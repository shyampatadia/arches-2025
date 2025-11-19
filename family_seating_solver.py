"""
Family seating optimization solver.

- Reads Excel with sheets:
    - "Adjacency_Preferences"
    - "Table_Preferences"
    - "Priority_Weights"
- Builds a CP-SAT model (Google OR-Tools) with:
    - Hard adjacency constraints for "Must" and "Cannot".
    - Hard capacity / assignment constraints.
    - Hard table "Cannot" constraints.
    - Soft objective terms for strong adjacency and table preferences,
      weighted by guest priority.
- Solves the model and writes all artefacts into a unique run directory:
    runs/YYYY-MM-DD_HH-MM-SS_run/

Outputs per run:
    - seating_plan.csv
    - adjacency_outcomes.csv
    - table_outcomes.csv
    - harmony_metrics.json
    - metadata.json
    - objective_progress.png
    - adjacency_result.png   (triangular heatmap with X on unsatisfied strong prefs)
"""

import os
import json
import time
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Utility: run directory
# ---------------------------------------------------------------------------

def create_run_directory(base_dir: str = "runs") -> str:
    """Create a unique directory for this run and return its path."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"{timestamp}_run")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Created run directory: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Data loading and normalization
# ---------------------------------------------------------------------------

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def load_inputs(excel_path: str,
                adj_sheet: str = "Adjacency_Preferences",
                table_sheet: str = "Table_Preferences",
                priority_sheet: str = "Priority_Weights"):
    """Load all three sheets and return (guests, adj_df, table_df, priority_dict)."""
    adj_df = pd.read_excel(excel_path, sheet_name=adj_sheet, index_col=0)
    tbl_df = pd.read_excel(excel_path, sheet_name=table_sheet, index_col=0)
    prio_df = pd.read_excel(excel_path, sheet_name=priority_sheet)

    guests = list(adj_df.index)
    tbl_df = tbl_df.reindex(index=guests)
    prio_df = prio_df.set_index("Guest").reindex(guests)

    priority = prio_df["Priority"].to_dict()

    return guests, adj_df, tbl_df, priority


# ---------------------------------------------------------------------------
# Chairs and neighbors
# ---------------------------------------------------------------------------

def build_chairs_and_neighbors(table_names):
    """
    Construct chairs and neighbor relationships.

    Assumes capacities from challenge:
        Big Round (0)      : 20 chairs, round
        Small Round (1)    :  6 chairs, round
        Balcony Rect (2)   :  8 chairs, line
        Kitchen Rect (3)   :  6 chairs, line
    """
    chairs = []  # list of (chair_id, table_name, position_index)
    chairs_by_table = {t: [] for t in table_names}
    round_tables = {t: False for t in table_names}

    def add_table(table_name, n_seats, is_round):
        nonlocal chairs
        for i in range(n_seats):
            cid = len(chairs)
            chairs.append((cid, table_name, i))
            chairs_by_table[table_name].append(cid)
        round_tables[table_name] = is_round

    # Use table_names order to map capacities
    if len(table_names) != 4:
        raise ValueError("Expected exactly 4 tables in Table_Preferences sheet.")

    add_table(table_names[0], 20, is_round=True)   # Big Round (Living Room)
    add_table(table_names[1], 6, is_round=True)    # Small Round (Living Room)
    add_table(table_names[2], 8, is_round=False)   # Balcony Rectangular
    add_table(table_names[3], 6, is_round=False)   # Kitchen Rectangular

    # Neighbor pairs across all tables
    neighbor_pairs = set()

    for t, cids in chairs_by_table.items():
        ids = list(cids)
        n = len(ids)
        if n <= 1:
            continue
        # linear neighbors
        for i in range(n - 1):
            c1, c2 = ids[i], ids[i + 1]
            neighbor_pairs.add(tuple(sorted((c1, c2))))
        # wrap-around if round
        if round_tables[t] and n > 2:
            neighbor_pairs.add(tuple(sorted((ids[0], ids[-1]))))

    neighbor_pairs = sorted(neighbor_pairs)

    chair_id_to_table = {cid: t for cid, t, pos in chairs}

    return chairs, chairs_by_table, round_tables, neighbor_pairs, chair_id_to_table


# ---------------------------------------------------------------------------
# Preference processing
# ---------------------------------------------------------------------------

def build_preferences(guests, adj_df, tbl_df):
    """
    From raw dataframes, build:
      - must_pairs, cannot_pairs (adjacency, unordered)
      - adj_soft_scores[(g1,g2)] -> score in {+2,-2}
      - table_score[g][table] -> score (strong prefs only)
      - table_cannot[g] -> set of tables that are 'cannot'
    """
    # adjacency
    must_pairs = set()
    cannot_pairs = set()
    adj_soft_scores = {}

    for g1 in guests:
        for g2 in guests:
            if g1 == g2:
                continue
            txt = normalize_text(adj_df.loc[g1, g2])
            if not txt:
                continue
            pair = tuple(sorted((g1, g2)))
            if txt == "must":
                must_pairs.add(pair)
            elif txt == "cannot":
                cannot_pairs.add(pair)
            elif txt == "strongly prefer":
                adj_soft_scores[pair] = 2
            elif txt == "strongly avoid":
                adj_soft_scores[pair] = -2
            # other levels treated as neutral

    # table preferences
    table_names = list(tbl_df.columns)
    table_score = defaultdict(dict)
    table_cannot = defaultdict(set)

    for g in guests:
        for t in table_names:
            txt = normalize_text(tbl_df.loc[g, t])
            if txt == "must":
                table_score[g][t] = 3
            elif txt == "strongly prefer":
                table_score[g][t] = 2
            elif txt == "strongly avoid":
                table_score[g][t] = -2
            elif txt == "cannot":
                table_score[g][t] = -5  # strong dislike for reporting
                table_cannot[g].add(t)  # but also hard constraint (cannot sit here)
            else:
                table_score[g][t] = 0

    return must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot, table_names


# ---------------------------------------------------------------------------
# Harmony monitor callback
# ---------------------------------------------------------------------------

class HarmonyMonitor(cp_model.CpSolverSolutionCallback):
    """Tracks objective value over time."""

    def __init__(self):
        super().__init__()
        self.times = []
        self.objectives = []
        self._start_time = time.time()

    def OnSolutionCallback(self):
        t = time.time() - self._start_time
        obj = self.ObjectiveValue()
        self.times.append(t)
        self.objectives.append(obj)
        print(f"[SOL] t={t:6.2f}s  objective={obj}")


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(guests,
                priority,
                chairs,
                chairs_by_table,
                neighbor_pairs,
                table_names,
                must_pairs,
                cannot_pairs,
                adj_soft_scores,
                table_score,
                table_cannot):
    """
    Build CP-SAT model and variables for seating problem.
    Returns: (model, x_vars, adj_soft_pair_vars, guest_index)
        - x_vars[(g_idx, chair_id)] -> BoolVar
        - adj_soft_pair_vars[(g1,g2)] -> BoolVar indicating they are adjacent
    """
    num_guests = len(guests)
    num_chairs = len(chairs)
    if num_guests != num_chairs:
        raise ValueError(f"Number of guests ({num_guests}) must equal total chairs ({num_chairs}).")

    guest_index = {g: i for i, g in enumerate(guests)}
    model = cp_model.CpModel()

    # Decision vars: x[g,c] = 1 if guest g sits in chair c
    x = {}
    for g in range(num_guests):
        for cid, tname, pos in chairs:
            x[g, cid] = model.NewBoolVar(f"x_g{g}_c{cid}")

    # Each guest in exactly one chair
    for g in range(num_guests):
        model.Add(sum(x[g, cid] for cid, _, _ in chairs) == 1)

    # Each chair has exactly one guest
    for cid, _, _ in chairs:
        model.Add(sum(x[g, cid] for g in range(num_guests)) == 1)

    # Hard table cannot constraints
    for gname, cant_tables in table_cannot.items():
        g = guest_index[gname]
        for t in cant_tables:
            for cid in chairs_by_table[t]:
                model.Add(x[g, cid] == 0)

    # Hard adjacency: cannot pairs (never adjacent)
    for (n1, n2) in cannot_pairs:
        g1, g2 = guest_index[n1], guest_index[n2]
        for (c1, c2) in neighbor_pairs:
            # forbid g1 at c1 with g2 at c2
            model.Add(x[g1, c1] + x[g2, c2] <= 1)
            # and g1 at c2 with g2 at c1
            model.Add(x[g1, c2] + x[g2, c1] <= 1)

    # Hard adjacency: must pairs (must be adjacent at least once)
    for (n1, n2) in must_pairs:
        g1, g2 = guest_index[n1], guest_index[n2]
        edge_vars = []
        for (c1, c2) in neighbor_pairs:
            z = model.NewBoolVar(f"must_{g1}_{g2}_c{c1}_{c2}")
            # one direction
            model.Add(z <= x[g1, c1])
            model.Add(z <= x[g2, c2])
            model.Add(z >= x[g1, c1] + x[g2, c2] - 1)
            # opposite direction
            z2 = model.NewBoolVar(f"must_{g1}_{g2}_c{c2}_{c1}")
            model.Add(z2 <= x[g1, c2])
            model.Add(z2 <= x[g2, c1])
            model.Add(z2 >= x[g1, c2] + x[g2, c1] - 1)
            edge_vars.extend([z, z2])
        # at least one adjacency
        model.Add(sum(edge_vars) >= 1)

    # Soft adjacency vars
    adj_soft_pair_vars = {}
    for (n1, n2), score in adj_soft_scores.items():
        g1, g2 = guest_index[n1], guest_index[n2]
        pair_var = model.NewBoolVar(f"adjsoft_{g1}_{g2}")
        adj_soft_pair_vars[(n1, n2)] = pair_var
        edge_vars = []
        for (c1, c2) in neighbor_pairs:
            # adjacency along this edge in one direction
            z = model.NewBoolVar(f"s_{g1}_{g2}_c{c1}_{c2}_a")
            model.Add(z <= x[g1, c1])
            model.Add(z <= x[g2, c2])
            model.Add(z >= x[g1, c1] + x[g2, c2] - 1)
            # and the opposite direction
            z2 = model.NewBoolVar(f"s_{g1}_{g2}_c{c2}_{c1}_b")
            model.Add(z2 <= x[g1, c2])
            model.Add(z2 <= x[g2, c1])
            model.Add(z2 >= x[g1, c2] + x[g2, c1] - 1)
            edge_vars.extend([z, z2])
        # pair_var == OR(edge_vars)
        model.Add(pair_var <= sum(edge_vars))
        for z in edge_vars:
            model.Add(z <= pair_var)

    # Objective: maximize harmony
    objective_terms = []

    # Table happiness
    for gname in guests:
        g = guest_index[gname]
        w = priority[gname]
        for t in table_names:
            s = table_score[gname][t]
            if s == 0:
                continue
            for cid in chairs_by_table[t]:
                objective_terms.append(int(w * s * 100) * x[g, cid])

    # Adjacency happiness (soft)
    for (n1, n2), s in adj_soft_scores.items():
        w = 0.5 * (priority[n1] + priority[n2])
        pair_var = adj_soft_pair_vars[(n1, n2)]
        coeff = int(w * s * 100)
        objective_terms.append(coeff * pair_var)

    model.Maximize(sum(objective_terms))

    return model, x, adj_soft_pair_vars, guest_index


# ---------------------------------------------------------------------------
# Harmony computation from a fixed seating plan
# ---------------------------------------------------------------------------

def compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables):
    """Rebuild neighbor pairs based on actual chair IDs and table roundness."""
    neighbor_pairs = set()
    for t, cids in chairs_by_table.items():
        ids = list(cids)
        n = len(ids)
        if n <= 1:
            continue
        for i in range(n - 1):
            neighbor_pairs.add(tuple(sorted((ids[i], ids[i+1]))))
        if round_tables[t] and n > 2:
            neighbor_pairs.add(tuple(sorted((ids[0], ids[-1]))))
    return neighbor_pairs


def compute_adjacency_set(seating_plan, neighbor_pairs):
    """Return set of frozenset({g1,g2}) that are adjacent under seating_plan."""
    chair_to_guest = dict(zip(seating_plan["ChairID"], seating_plan["Guest"]))
    adjacent_pairs = set()
    for (c1, c2) in neighbor_pairs:
        g1 = chair_to_guest[c1]
        g2 = chair_to_guest[c2]
        if g1 == g2:
            continue
        adjacent_pairs.add(frozenset((g1, g2)))
    return adjacent_pairs


def compute_harmony(guests,
                    seating_plan,
                    priority,
                    adj_soft_scores,
                    table_score,
                    chairs_by_table,
                    round_tables):
    """Compute overall harmony components from a solved seating plan."""
    neighbor_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, neighbor_pairs)

    def are_adjacent(g1, g2):
        return frozenset((g1, g2)) in adj_set

    # guest -> table
    assigned_table = seating_plan.set_index("Guest")["Table"].to_dict()

    # adjacency harmony
    adj_harmony = 0.0
    max_adj_harmony = 0.0
    for (g1, g2), s in adj_soft_scores.items():
        w = 0.5 * (priority[g1] + priority[g2])
        adj = are_adjacent(g1, g2)
        if s > 0:
            if adj:
                adj_harmony += w * s
            max_adj_harmony += w * s
        elif s < 0:
            if not adj:
                adj_harmony += w * (-s)  # avoiding negative is good
            max_adj_harmony += w * (-s)

    # table harmony
    tbl_harmony = 0.0
    max_tbl_harmony = 0.0
    for g in guests:
        t_assigned = assigned_table[g]
        w = priority[g]
        s_assigned = table_score[g][t_assigned]
        tbl_harmony += w * s_assigned
        best = max(table_score[g].values()) if table_score[g] else 0
        max_tbl_harmony += w * best

    overall = adj_harmony + tbl_harmony

    metrics = {
        "overall_harmony": overall,
        "adj_harmony": adj_harmony,
        "tbl_harmony": tbl_harmony,
        "max_adj_harmony": max_adj_harmony,
        "max_tbl_harmony": max_tbl_harmony,
        "adj_satisfaction_pct": (adj_harmony / max_adj_harmony) if max_adj_harmony > 0 else 1.0,
        "tbl_satisfaction_pct": (tbl_harmony / max_tbl_harmony) if max_tbl_harmony > 0 else 1.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Visualization: adjacency result heatmap with unsatisfied strong prefs
# ---------------------------------------------------------------------------

def make_adjacency_result_heatmap(guests,
                                  adj_df,
                                  seating_plan,
                                  chairs_by_table,
                                  round_tables,
                                  run_dir,
                                  filename="adjacency_result.png"):
    """
    Create triangular adjacency heatmap of strong preferences, overlaying
    an 'X' where a strong preference (Must / Cannot / Strongly prefer / Strongly avoid)
    is NOT satisfied by the seating plan.
    """
    n = len(guests)
    idx = {g: i for i, g in enumerate(guests)}

    code_map = {
        "cannot": -2,
        "strongly avoid": -1,
        "strongly prefer": 1,
        "must": 2,
    }

    # preference matrix
    pref_codes = np.zeros((n, n), dtype=float)
    strong_type = np.empty((n, n), dtype=object)
    strong_type[:] = ""

    for g1 in guests:
        for g2 in guests:
            if g1 == g2:
                continue
            txt = normalize_text(adj_df.loc[g1, g2])
            if txt in code_map:
                i, j = idx[g1], idx[g2]
                pref_codes[i, j] = code_map[txt]
                strong_type[i, j] = txt

    neighbor_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, neighbor_pairs)

    def are_adjacent(g1, g2):
        return frozenset((g1, g2)) in adj_set

    # unsatisfied matrix
    unsat = np.zeros((n, n), dtype=int)
    for g1 in guests:
        for g2 in guests:
            if g1 == g2:
                continue
            i, j = idx[g1], idx[g2]
            txt = strong_type[i, j]
            if not txt:
                continue
            adj = are_adjacent(g1, g2)
            if txt in ("must", "strongly prefer"):
                satisfied = adj
            else:  # "cannot" or "strongly avoid"
                satisfied = not adj
            if not satisfied:
                unsat[i, j] = 1

    # mask to show only upper triangle (inverted vs your first plot)
    mask_triangle = np.tril(np.ones_like(pref_codes, dtype=bool), k=-1)

    sns.set_theme(style="white")
    colors = [
            "#4b0082",  # -2 Cannot
            "#b22222",  # -1 Strongly avoid
            "#f0f0f0",  #  0 Neutral
            "#228b22",  #  1 Strongly prefer
            "#006400",  #  2 Must
    ]
    cmap = ListedColormap(colors)
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    hm = sns.heatmap(
        pref_codes,
        cmap=cmap,
        norm=norm,
        mask=mask_triangle,
        annot=False,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
        ax=ax,
        cbar_kws={"ticks": [-2, -1, 0, 1, 2]},
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticklabels(["Cannot", "Strongly avoid", "Neutral", "Strongly prefer", "Must"])
    cbar.set_label("Adjacency preference type", rotation=90)

    # overlay X for unsatisfied strong prefs
    for i in range(n):
        for j in range(n):
            if mask_triangle[i, j]:
                continue
            if unsat[i, j] == 1 and pref_codes[i, j] != 0:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                )

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(guests, rotation=45, ha="right")
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(guests, rotation=0)

    ax.set_xlabel("Person on the right")
    ax.set_ylabel("Person")
    ax.set_title("Adjacency preferences vs actual seating\n(X = strong preference not satisfied)")

    plt.tight_layout()
    out_path = os.path.join(run_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved adjacency comparison heatmap to {out_path}")


# ---------------------------------------------------------------------------
# Objective progress plot
# ---------------------------------------------------------------------------

def plot_objective_progress(monitor: HarmonyMonitor, run_dir: str, filename="objective_progress.png"):
    if not monitor.times:
        print("[WARN] No intermediate solutions recorded by HarmonyMonitor.")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(monitor.times, monitor.objectives, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Objective value")
    plt.title("Objective progress during search")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(run_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved objective progress plot to {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def solve_seating(excel_path: str,
                  max_time_seconds: float = 60.0,
                  base_run_dir: str = "runs"):
    """Main entry point: solve, evaluate and write a full run directory."""
    run_dir = create_run_directory(base_run_dir)

    # 1) Load
    guests, adj_df, tbl_df, priority = load_inputs(excel_path)
    must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot, table_names = \
        build_preferences(guests, adj_df, tbl_df)
    chairs, chairs_by_table, round_tables, neighbor_pairs, chair_id_to_table = \
        build_chairs_and_neighbors(table_names)

    # 2) Build model
    model, x, adj_soft_pair_vars, guest_index = build_model(
        guests, priority,
        chairs, chairs_by_table, neighbor_pairs, table_names,
        must_pairs, cannot_pairs, adj_soft_scores,
        table_score, table_cannot
    )

    # 3) Solve with monitor
    monitor = HarmonyMonitor()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 0  # use all cores

    print(f"[INFO] Starting solver for up to {max_time_seconds} seconds...")
    status = solver.Solve(model, monitor)
    print(f"[INFO] Solver finished with status: {solver.StatusName(status)}")
    best_objective = solver.ObjectiveValue()
    print(f"[INFO] Best objective value: {best_objective}")

    # 4) Extract seating plan
    assignments = []
    for gname in guests:
        g = guest_index[gname]
        for cid, tname, pos in chairs:
            if solver.Value(x[g, cid]) == 1:
                assignments.append({
                    "Guest": gname,
                    "Table": tname,
                    "ChairID": cid,
                    "SeatIndexAtTable": pos,
                    "Priority": priority[gname],
                })
                break

    seating_plan = pd.DataFrame(assignments)
    seating_path = os.path.join(run_dir, "seating_plan.csv")
    seating_plan.to_csv(seating_path, index=False)
    print(f"[INFO] Saved seating plan to {seating_path}")

    # 5) Build adjacency outcomes dataframe for dashboard
    nb_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, nb_pairs)

    def are_adjacent(g1, g2):
        return frozenset((g1, g2)) in adj_set

    adj_rows = []
    # soft adjacency pairs
    for (g1, g2), s in adj_soft_scores.items():
        adj = are_adjacent(g1, g2)
        w = 0.5 * (priority[g1] + priority[g2])
        adj_rows.append({
            "Guest1": g1,
            "Guest2": g2,
            "PreferenceType": "Strongly prefer" if s > 0 else "Strongly avoid",
            "PreferenceScore": s,
            "PriorityAvg": w,
            "AreAdjacent": int(adj),
        })
    # must / cannot pairs
    for (g1, g2) in must_pairs:
        adj = are_adjacent(g1, g2)
        w = 0.5 * (priority[g1] + priority[g2])
        adj_rows.append({
            "Guest1": g1,
            "Guest2": g2,
            "PreferenceType": "Must",
            "PreferenceScore": None,
            "PriorityAvg": w,
            "AreAdjacent": int(adj),
            "HardSatisfied": int(adj),
        })
    for (g1, g2) in cannot_pairs:
        adj = are_adjacent(g1, g2)
        w = 0.5 * (priority[g1] + priority[g2])
        adj_rows.append({
            "Guest1": g1,
            "Guest2": g2,
            "PreferenceType": "Cannot",
            "PreferenceScore": None,
            "PriorityAvg": w,
            "AreAdjacent": int(adj),
            "HardSatisfied": int(not adj),
        })

    adjacency_outcomes = pd.DataFrame(adj_rows)
    adj_path = os.path.join(run_dir, "adjacency_outcomes.csv")
    adjacency_outcomes.to_csv(adj_path, index=False)
    print(f"[INFO] Saved adjacency outcomes to {adj_path}")

    # 6) Table outcomes dataframe
    tbl_rows = []
    assigned_table = seating_plan.set_index("Guest")["Table"].to_dict()
    for g in guests:
        t_assigned = assigned_table[g]
        for t in table_names:
            s = table_score[g][t]
            tbl_rows.append({
                "Guest": g,
                "Table": t,
                "AssignedHere": int(t == t_assigned),
                "PreferenceScore": s,
                "Priority": priority[g],
            })
    table_outcomes = pd.DataFrame(tbl_rows)
    tbl_path = os.path.join(run_dir, "table_outcomes.csv")
    table_outcomes.to_csv(tbl_path, index=False)
    print(f"[INFO] Saved table outcomes to {tbl_path}")

    # 7) Harmony metrics
    metrics = compute_harmony(
        guests, seating_plan, priority,
        adj_soft_scores, table_score,
        chairs_by_table, round_tables
    )
    metrics["raw_objective_value"] = best_objective

    metrics_path = os.path.join(run_dir, "harmony_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved harmony metrics to {metrics_path}")

    # 8) Metadata
    meta = {
        "run_dir": run_dir,
        "excel_path": excel_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "solver_status": solver.StatusName(status),
        "max_time_seconds": max_time_seconds,
        "num_guests": len(guests),
        "num_tables": len(table_names),
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[INFO] Saved run metadata to {meta_path}")

    # 9) Plots
    plot_objective_progress(monitor, run_dir)
    make_adjacency_result_heatmap(
        guests, adj_df, seating_plan,
        chairs_by_table, round_tables,
        run_dir
    )

    print(f"\n[DONE] Run complete. All outputs written to:\n  {run_dir}\n")
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Family seating optimization solver.")
    parser.add_argument("excel_path", help="Path to Excel file with preferences.")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Maximum solve time in seconds (default: 60).")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Base directory for run outputs (default: runs)")
    args = parser.parse_args()
    solve_seating(args.excel_path, max_time_seconds=args.time, base_run_dir=args.runs_dir)
