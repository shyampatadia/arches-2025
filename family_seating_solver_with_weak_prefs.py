"""
Family seating optimization solver with CONSTRAINT SATISFACTION RATIOS.

Enhanced to support:
- Satisfaction ratios for adjacency and table constraints
- Detailed breakdown by constraint type (Must, Cannot, Strongly Prefer, Strongly Avoid, Prefer, Avoid)
- CLI flag to include weak preferences (Prefer/Avoid) in optimization
- Visual dashboard with ratio metrics
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
# Load inputs
# ---------------------------------------------------------------------------

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def load_inputs(
    excel_path: str,
    adj_sheet: str = "Adjacency_Preferences",
    table_sheet: str = "Table_Preferences",
    priority_sheet: str = "Priority_Weights"
):
    """Load all three sheets and return guests, adjdf, tabledf, priority dict."""
    adj_df = pd.read_excel(excel_path, sheet_name=adj_sheet, index_col=0)
    tbl_df = pd.read_excel(excel_path, sheet_name=table_sheet, index_col=0)
    prio_df = pd.read_excel(excel_path, sheet_name=priority_sheet)

    guests = list(adj_df.index)
    tbl_df = tbl_df.reindex(index=guests)
    prio_df = prio_df.set_index("Guest").reindex(guests)
    priority = prio_df["Priority"].to_dict()

    return guests, adj_df, tbl_df, priority


# ---------------------------------------------------------------------------
# Build chairs and neighbors
# ---------------------------------------------------------------------------

def build_chairs_and_neighbors(table_names):
    """
    Construct chairs and neighbor relationships.
    Assumes capacities from challenge:
      Big Round (0)    = 20 chairs, round
      Small Round (1)  = 6 chairs, round
      Balcony Rect (2) = 8 chairs, line
      Kitchen Rect (3) = 6 chairs, line
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
    add_table(table_names[2], 8, is_round=False)   # Balcony (Rectangular)
    add_table(table_names[3], 6, is_round=False)   # Kitchen (Rectangular)

    # Neighbor pairs across all tables
    neighbor_pairs = set()
    for t, cids in chairs_by_table.items():
        ids = list(cids)
        n = len(ids)
        if n < 2:
            continue
        # linear neighbors
        for i in range(n - 1):
            c1, c2 = ids[i], ids[i+1]
            neighbor_pairs.add(tuple(sorted([c1, c2])))
        # wrap-around if round
        if round_tables[t] and n >= 2:
            neighbor_pairs.add(tuple(sorted([ids[0], ids[-1]])))

    neighbor_pairs = sorted(neighbor_pairs)
    chair_id_to_table = {cid: t for (cid, t, pos) in chairs}

    return chairs, chairs_by_table, round_tables, neighbor_pairs, chair_id_to_table


# ---------------------------------------------------------------------------
# Build preferences (ENHANCED with weak preferences support)
# ---------------------------------------------------------------------------

def build_preferences(guests, adj_df, tbl_df, include_weak=False):
    """
    From raw dataframes, build:
      - must_pairs, cannot_pairs (adjacency, unordered)
      - adj_soft_scores[(g1,g2)] = score in {2, 1, -1, -2}
      - table_score[g][table] = score (all prefs if include_weak, strong only otherwise)
      - table_cannot[g] = set of tables that are "cannot"
    
    If include_weak=True:
      - Adjacency: "prefer" → +1, "avoid" → -1
      - Table: "prefer" → +1, "avoid" → -1
    """
    must_pairs = set()
    cannot_pairs = set()
    adj_soft_scores = {}

    # adjacency
    for g1 in guests:
        for g2 in guests:
            if g1 >= g2:
                continue
            txt = normalize_text(adj_df.loc[g1, g2])
            if not txt:
                continue
            pair = tuple(sorted([g1, g2]))
            if txt == "must":
                must_pairs.add(pair)
            elif txt == "cannot":
                cannot_pairs.add(pair)
            elif txt == "strongly prefer":
                adj_soft_scores[pair] = 2
            elif txt == "strongly avoid":
                adj_soft_scores[pair] = -2
            elif include_weak:
                if txt == "prefer":
                    adj_soft_scores[pair] = 1
                elif txt == "avoid":
                    adj_soft_scores[pair] = -1

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
                table_cannot[g].add(t)  # but also hard constraint: cannot sit here
            elif include_weak:
                if txt == "prefer":
                    table_score[g][t] = 1
                elif txt == "avoid":
                    table_score[g][t] = -1
                else:
                    table_score[g][t] = 0
            else:
                table_score[g][t] = 0

    return must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot, table_names


# ---------------------------------------------------------------------------
# Harmony Monitor
# ---------------------------------------------------------------------------

class HarmonyMonitor(cp_model.CpSolverSolutionCallback):
    """Tracks objective value over time."""
    def __init__(self):
        super().__init__()
        self.times = []
        self.objectives = []
        self.start_time = time.time()

    def on_solution_callback(self):
        t = time.time() - self.start_time
        obj = self.ObjectiveValue()
        self.times.append(t)
        self.objectives.append(obj)
        print(f"[SOL] t={t:6.2f}s  objective={obj}")


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(
    guests, priority, chairs, chairs_by_table, neighbor_pairs,
    table_names, must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot
):
    """
    Build CP-SAT model and variables for seating problem.
    Returns (model, x_vars, adj_soft_pair_vars, guest_index).
      - x_vars[g_idx, chair_id] = BoolVar
      - adj_soft_pair_vars[(g1,g2)] = BoolVar indicating they are adjacent
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
        for (cid, tname, pos) in chairs:
            x[g, cid] = model.NewBoolVar(f"x[g={g}][c={cid}]")

    # Each guest in exactly one chair
    for g in range(num_guests):
        model.Add(sum(x[g, cid] for (cid, _, _) in chairs) == 1)

    # Each chair has exactly one guest
    for (cid, _, _) in chairs:
        model.Add(sum(x[g, cid] for g in range(num_guests)) == 1)

    # Hard table "cannot" constraints
    for gname, cant_tables in table_cannot.items():
        g = guest_index[gname]
        for t in cant_tables:
            for cid in chairs_by_table[t]:
                model.Add(x[g, cid] == 0)

    # Hard adjacency: "cannot" pairs never adjacent
    for (n1, n2) in cannot_pairs:
        g1, g2 = guest_index[n1], guest_index[n2]
        for (c1, c2) in neighbor_pairs:
            # forbid g1 at c1 with g2 at c2
            model.Add(x[g1, c1] + x[g2, c2] <= 1)
            # and g1 at c2 with g2 at c1
            model.Add(x[g1, c2] + x[g2, c1] <= 1)

    # Hard adjacency: "must" pairs must be adjacent at least once
    for (n1, n2) in must_pairs:
        g1, g2 = guest_index[n1], guest_index[n2]
        edge_vars = []
        for (c1, c2) in neighbor_pairs:
            # one direction
            z = model.NewBoolVar(f"must[{g1},{g2}][c={c1},{c2}]")
            model.Add(z <= x[g1, c1])
            model.Add(z <= x[g2, c2])
            model.Add(z >= x[g1, c1] + x[g2, c2] - 1)
            # opposite direction
            z2 = model.NewBoolVar(f"must[{g1},{g2}][c={c2},{c1}]")
            model.Add(z2 <= x[g1, c2])
            model.Add(z2 <= x[g2, c1])
            model.Add(z2 >= x[g1, c2] + x[g2, c1] - 1)
            edge_vars.extend([z, z2])
        # at least one adjacency
        model.Add(sum(edge_vars) >= 1)

    # Soft adjacency vars
    adj_soft_pair_vars = {}
    for (n1, n2, score) in [(n1, n2, s) for ((n1, n2), s) in adj_soft_scores.items()]:
        g1, g2 = guest_index[n1], guest_index[n2]
        pair_var = model.NewBoolVar(f"adj_soft[{g1},{g2}]")
        adj_soft_pair_vars[(n1, n2)] = pair_var

        edge_vars = []
        for (c1, c2) in neighbor_pairs:
            # adjacency along this edge in one direction
            z = model.NewBoolVar(f"s[{g1},{g2}][c={c1},{c2}]a")
            model.Add(z <= x[g1, c1])
            model.Add(z <= x[g2, c2])
            model.Add(z >= x[g1, c1] + x[g2, c2] - 1)
            # and the opposite direction
            z2 = model.NewBoolVar(f"s[{g1},{g2}][c={c2},{c1}]b")
            model.Add(z2 <= x[g1, c2])
            model.Add(z2 <= x[g2, c1])
            model.Add(z2 >= x[g1, c2] + x[g2, c1] - 1)
            edge_vars.extend([z, z2])
        # pair_var = OR(edge_vars)
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
    for (n1, n2, s) in [(n1, n2, s) for ((n1, n2), s) in adj_soft_scores.items()]:
        w = 0.5 * (priority[n1] + priority[n2])
        pair_var = adj_soft_pair_vars[(n1, n2)]
        coeff = int(w * s * 100)
        objective_terms.append(coeff * pair_var)

    model.Maximize(sum(objective_terms))

    return model, x, adj_soft_pair_vars, guest_index


# ---------------------------------------------------------------------------
# Helper: compute neighbor pairs from layout
# ---------------------------------------------------------------------------

def compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables):
    """Rebuild neighbor pairs based on actual chair IDs and table roundness."""
    neighbor_pairs = set()
    for t, cids in chairs_by_table.items():
        ids = list(cids)
        n = len(ids)
        if n < 2:
            continue
        for i in range(n - 1):
            neighbor_pairs.add(tuple(sorted([ids[i], ids[i+1]])))
        if round_tables[t] and n >= 2:
            neighbor_pairs.add(tuple(sorted([ids[0], ids[-1]])))
    return neighbor_pairs


def compute_adjacency_set(seating_plan, neighbor_pairs):
    """Return set of frozenset({g1,g2}) that are adjacent under seating_plan."""
    chair_to_guest = dict(zip(seating_plan["ChairID"], seating_plan["Guest"]))
    adjacent_pairs = set()
    for (c1, c2) in neighbor_pairs:
        g1 = chair_to_guest[c1]
        g2 = chair_to_guest[c2]
        if g1 != g2:
            adjacent_pairs.add(frozenset({g1, g2}))
    return adjacent_pairs


# ---------------------------------------------------------------------------
# Compute harmony
# ---------------------------------------------------------------------------

def compute_harmony(
    guests, seating_plan, priority, adj_soft_scores, table_score, chairs_by_table, round_tables
):
    """Compute overall harmony components from a solved seating plan."""
    neighbor_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, neighbor_pairs)

    def are_adjacent(g1, g2):
        return frozenset({g1, g2}) in adj_set

    assigned_table = seating_plan.set_index("Guest")["Table"].to_dict()  # guest -> table

    # adjacency harmony
    adj_harmony = 0.0
    max_adj_harmony = 0.0
    for (g1, g2, s) in [(g1, g2, s) for ((g1, g2), s) in adj_soft_scores.items()]:
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
# NEW: Compute detailed satisfaction ratios (ENHANCED with weak preferences)
# ---------------------------------------------------------------------------

def compute_satisfaction_ratios(
    guests, seating_plan, must_pairs, cannot_pairs, adj_soft_scores,
    table_score, table_cannot, chairs_by_table, round_tables, priority, include_weak=False
):
    """
    Calculate satisfaction ratios for all constraint types.
    Returns a dictionary with detailed breakdowns.
    """
    neighbor_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, neighbor_pairs)
    assigned_table = seating_plan.set_index("Guest")["Table"].to_dict()

    def are_adjacent(g1, g2):
        return frozenset({g1, g2}) in adj_set

    # Initialize counters with weak preferences if enabled
    pref_types = ["must", "cannot", "strongly_prefer", "strongly_avoid"]
    if include_weak:
        pref_types.extend(["prefer", "avoid"])
    
    ratios = {
        "adjacency": {pt: {"satisfied": 0, "total": 0, "ratio": 0.0} for pt in pref_types},
        "table": {pt: {"satisfied": 0, "total": 0, "ratio": 0.0} for pt in pref_types},
    }

    # Adjacency: Must pairs
    for (g1, g2) in must_pairs:
        ratios["adjacency"]["must"]["total"] += 1
        if are_adjacent(g1, g2):
            ratios["adjacency"]["must"]["satisfied"] += 1

    # Adjacency: Cannot pairs
    for (g1, g2) in cannot_pairs:
        ratios["adjacency"]["cannot"]["total"] += 1
        if not are_adjacent(g1, g2):
            ratios["adjacency"]["cannot"]["satisfied"] += 1

    # Adjacency: Soft preferences
    for (g1, g2), score in adj_soft_scores.items():
        if score == 2:  # Strongly prefer
            ratios["adjacency"]["strongly_prefer"]["total"] += 1
            if are_adjacent(g1, g2):
                ratios["adjacency"]["strongly_prefer"]["satisfied"] += 1
        elif score == -2:  # Strongly avoid
            ratios["adjacency"]["strongly_avoid"]["total"] += 1
            if not are_adjacent(g1, g2):
                ratios["adjacency"]["strongly_avoid"]["satisfied"] += 1
        elif include_weak:
            if score == 1:  # Prefer
                ratios["adjacency"]["prefer"]["total"] += 1
                if are_adjacent(g1, g2):
                    ratios["adjacency"]["prefer"]["satisfied"] += 1
            elif score == -1:  # Avoid
                ratios["adjacency"]["avoid"]["total"] += 1
                if not are_adjacent(g1, g2):
                    ratios["adjacency"]["avoid"]["satisfied"] += 1

    # Table preferences
    for g in guests:
        t_assigned = assigned_table[g]
        for t, score in table_score[g].items():
            if score == 3:  # Must
                ratios["table"]["must"]["total"] += 1
                if t == t_assigned:
                    ratios["table"]["must"]["satisfied"] += 1
            elif score == -5 or t in table_cannot[g]:  # Cannot
                ratios["table"]["cannot"]["total"] += 1
                if t != t_assigned:
                    ratios["table"]["cannot"]["satisfied"] += 1
            elif score == 2:  # Strongly prefer
                ratios["table"]["strongly_prefer"]["total"] += 1
                if t == t_assigned:
                    ratios["table"]["strongly_prefer"]["satisfied"] += 1
            elif score == -2:  # Strongly avoid
                ratios["table"]["strongly_avoid"]["total"] += 1
                if t != t_assigned:
                    ratios["table"]["strongly_avoid"]["satisfied"] += 1
            elif include_weak:
                if score == 1:  # Prefer
                    ratios["table"]["prefer"]["total"] += 1
                    if t == t_assigned:
                        ratios["table"]["prefer"]["satisfied"] += 1
                elif score == -1:  # Avoid
                    ratios["table"]["avoid"]["total"] += 1
                    if t != t_assigned:
                        ratios["table"]["avoid"]["satisfied"] += 1

    # Calculate ratios
    for constraint_type in ["adjacency", "table"]:
        for pref_type in ratios[constraint_type].keys():
            total = ratios[constraint_type][pref_type]["total"]
            satisfied = ratios[constraint_type][pref_type]["satisfied"]
            if total > 0:
                ratios[constraint_type][pref_type]["ratio"] = satisfied / total
            else:
                ratios[constraint_type][pref_type]["ratio"] = None  # No constraints of this type

    # Overall satisfaction metrics
    total_constraints = sum(
        ratios[ct][pt]["total"]
        for ct in ["adjacency", "table"]
        for pt in ratios[ct].keys()
    )
    total_satisfied = sum(
        ratios[ct][pt]["satisfied"]
        for ct in ["adjacency", "table"]
        for pt in ratios[ct].keys()
    )

    ratios["overall"] = {
        "satisfied": total_satisfied,
        "total": total_constraints,
        "ratio": total_satisfied / total_constraints if total_constraints > 0 else 1.0
    }

    return ratios


# ---------------------------------------------------------------------------
# NEW: Create satisfaction ratio visualization (ENHANCED for weak preferences)
# ---------------------------------------------------------------------------

def plot_satisfaction_ratios(ratios, run_dir, include_weak=False, filename="satisfaction_ratios.png"):
    """Create a comprehensive visualization of satisfaction ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Constraint Satisfaction Analysis' + (' (Including Weak Preferences)' if include_weak else ''),
                 fontsize=16, fontweight='bold')

    # Color scheme
    colors = {
        "must": "#006400",          # Dark green
        "cannot": "#b22222",        # Dark red
        "strongly_prefer": "#228b22",  # Green
        "strongly_avoid": "#ff6347",   # Tomato
        "prefer": "#90ee90",        # Light green
        "avoid": "#ffb6c1",         # Light red
    }

    # 1. Overall satisfaction gauge
    ax = axes[0, 0]
    overall = ratios["overall"]
    ratio = overall["ratio"] * 100
    
    wedges, texts, autotexts = ax.pie(
        [overall["satisfied"], overall["total"] - overall["satisfied"]],
        labels=['Satisfied', 'Unsatisfied'],
        colors=['#32b896', '#e0e0e0'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    ax.set_title(f'Overall Satisfaction\n{overall["satisfied"]}/{overall["total"]} constraints', 
                 fontsize=13, fontweight='bold', pad=20)

    # 2. Adjacency constraints breakdown
    ax = axes[0, 1]
    adj_data = ratios["adjacency"]
    pref_types = ["must", "cannot", "strongly_prefer", "strongly_avoid"]
    if include_weak:
        pref_types.extend(["prefer", "avoid"])
    
    categories = []
    satisfied = []
    total = []
    ratio_vals = []
    bar_colors = []
    
    for pref_type in pref_types:
        if adj_data[pref_type]["total"] > 0:
            categories.append(pref_type.replace("_", "\n"))
            satisfied.append(adj_data[pref_type]["satisfied"])
            total.append(adj_data[pref_type]["total"])
            ratio_vals.append(adj_data[pref_type]["ratio"] * 100)
            bar_colors.append(colors.get(pref_type, '#999999'))
    
    if categories:
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, satisfied, width, label='Satisfied', 
                      color=bar_colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, [total[i] - satisfied[i] for i in range(len(total))], 
                       width, label='Unsatisfied', color='#e0e0e0', alpha=0.8)
        
        # Add ratio labels on top
        for i, (bar, ratio_val) in enumerate(zip(bars1, ratio_vals)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{ratio_val:.0f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Adjacency Constraints', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No adjacency constraints', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    # 3. Table constraints breakdown
    ax = axes[1, 0]
    tbl_data = ratios["table"]
    categories = []
    satisfied = []
    total = []
    ratio_vals = []
    bar_colors = []
    
    for pref_type in pref_types:
        if tbl_data[pref_type]["total"] > 0:
            categories.append(pref_type.replace("_", "\n"))
            satisfied.append(tbl_data[pref_type]["satisfied"])
            total.append(tbl_data[pref_type]["total"])
            ratio_vals.append(tbl_data[pref_type]["ratio"] * 100)
            bar_colors.append(colors.get(pref_type, '#999999'))
    
    if categories:
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, satisfied, width, label='Satisfied', 
                      color=bar_colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, [total[i] - satisfied[i] for i in range(len(total))], 
                       width, label='Unsatisfied', color='#e0e0e0', alpha=0.8)
        
        # Add ratio labels on top
        for i, (bar, ratio_val) in enumerate(zip(bars1, ratio_vals)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{ratio_val:.0f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Table Constraints', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No table constraints', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    summary_data.append(['CONSTRAINT TYPE', 'SAT', 'TOTAL', 'RATIO'])
    summary_data.append(['─' * 18, '─' * 5, '─' * 6, '─' * 8])
    
    # Adjacency summary
    for pref_type in pref_types:
        data = adj_data[pref_type]
        if data["total"] > 0:
            label = f"Adj: {pref_type.replace('_', ' ').title()}"
            summary_data.append([
                label,
                str(data["satisfied"]),
                str(data["total"]),
                f"{data['ratio']*100:.1f}%"
            ])
    
    summary_data.append(['', '', '', ''])
    
    # Table summary
    for pref_type in pref_types:
        data = tbl_data[pref_type]
        if data["total"] > 0:
            label = f"Tbl: {pref_type.replace('_', ' ').title()}"
            summary_data.append([
                label,
                str(data["satisfied"]),
                str(data["total"]),
                f"{data['ratio']*100:.1f}%"
            ])
    
    summary_data.append(['═' * 18, '═' * 5, '═' * 6, '═' * 8])
    summary_data.append([
        'OVERALL',
        str(overall["satisfied"]),
        str(overall["total"]),
        f"{overall['ratio']*100:.1f}%"
    ])
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.15, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#32b896')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style overall row
    for i in range(4):
        table[(len(summary_data)-1, i)].set_facecolor('#e8f5e9')
        table[(len(summary_data)-1, i)].set_text_props(weight='bold')
    
    ax.set_title('Detailed Breakdown', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = os.path.join(run_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved satisfaction ratios plot to: {out_path}")


# ---------------------------------------------------------------------------
# Make adjacency result heatmap (ENHANCED with ratio annotation)
# ---------------------------------------------------------------------------

def make_adjacency_result_heatmap(
    guests, adj_df, seating_plan, chairs_by_table, round_tables, 
    ratios, run_dir, include_weak=False, filename="adjacency_result.png"
):
    """
    Create triangular adjacency heatmap of preferences,
    overlaying an X where a preference is NOT satisfied.
    Also adds satisfaction ratio text.
    """
    n = len(guests)
    idx = {g: i for i, g in enumerate(guests)}

    # Code map - extended for weak preferences
    code_map = {
        "cannot": -3,
        "strongly avoid": -2,
        "avoid": -1,
        "prefer": 1,
        "strongly prefer": 2,
        "must": 3,
    }

    # preference matrix
    pref_codes = np.zeros((n, n), dtype=float)
    strong_type = np.empty((n, n), dtype=object)
    for g1 in guests:
        for g2 in guests:
            if g1 >= g2:
                continue
            txt = normalize_text(adj_df.loc[g1, g2])
            if txt in code_map:
                i, j = idx[g1], idx[g2]
                pref_codes[i, j] = code_map[txt]
                strong_type[i, j] = txt

    neighbor_pairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, neighbor_pairs)

    def are_adjacent(g1, g2):
        return frozenset({g1, g2}) in adj_set

    # unsatisfied matrix
    unsat = np.zeros((n, n), dtype=int)
    for g1 in guests:
        for g2 in guests:
            if g1 >= g2:
                continue
            i, j = idx[g1], idx[g2]
            txt = strong_type[i, j]
            if not txt:
                continue
            # Only mark as unsatisfied if include_weak or if it's a strong preference
            if not include_weak and txt in ["prefer", "avoid"]:
                continue
            adj = are_adjacent(g1, g2)
            if txt in ["must", "strongly prefer", "prefer"]:
                satisfied = adj
            else:  # "cannot", "strongly avoid", or "avoid"
                satisfied = not adj
            if not satisfied:
                unsat[i, j] = 1

    # mask to show only upper triangle
    mask_triangle = np.tril(np.ones_like(pref_codes, dtype=bool), k=-1)

    sns.set_theme(style="white")

    # Color scheme - extended for weak preferences
    colors = {
        -3: "#4b0082",  # Cannot
        -2: "#b22222",  # Strongly avoid
        -1: "#ffb6c1",  # Avoid (light red)
        0: "#f0f0f0",   # Neutral
        1: "#90ee90",   # Prefer (light green)
        2: "#228b22",   # Strongly prefer
        3: "#006400",   # Must
    }
    cmap = ListedColormap([colors[k] for k in sorted(colors.keys())])
    bounds = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(14, 11))

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
        cbar_kws={"ticks": [-3, -2, -1, 0, 1, 2, 3]}
    )
    cbar = hm.collections[0].colorbar
    if include_weak:
        cbar.set_ticklabels(["Cannot", "Strong avoid", "Avoid", "Neutral", 
                            "Prefer", "Strong prefer", "Must"])
    else:
        cbar.set_ticklabels(["Cannot", "Strong avoid", "Neutral", "Neutral", 
                            "Neutral", "Strong prefer", "Must"])
    cbar.set_label("Adjacency preference type", rotation=90)

    # overlay X for unsatisfied prefs
    for i in range(n):
        for j in range(n):
            if mask_triangle[i, j]:
                continue
            if unsat[i, j] == 1 and pref_codes[i, j] != 0:
                ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center',
                       fontsize=9, color='black', fontweight='bold')

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(guests, rotation=45, ha='right')
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(guests, rotation=0)
    ax.set_xlabel("Person on the right")
    ax.set_ylabel("Person")
    
    # Add satisfaction ratio to title
    adj_ratios = ratios["adjacency"]
    total_adj_constraints = sum(adj_ratios[pt]["total"] for pt in adj_ratios.keys())
    total_adj_satisfied = sum(adj_ratios[pt]["satisfied"] for pt in adj_ratios.keys())
    adj_ratio_pct = (total_adj_satisfied / total_adj_constraints * 100) if total_adj_constraints > 0 else 100.0
    
    title_suffix = " (including weak preferences)" if include_weak else ""
    ax.set_title(
        f"Adjacency Preferences vs Actual Seating{title_suffix}\n"
        f"✓ Satisfied: {total_adj_satisfied}/{total_adj_constraints} ({adj_ratio_pct:.1f}%) | "
        f"✗ marks unsatisfied constraints",
        fontsize=13,
        fontweight='bold',
        pad=15
    )

    plt.tight_layout()
    out_path = os.path.join(run_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved adjacency comparison heatmap to: {out_path}")


# ---------------------------------------------------------------------------
# Plot objective progress
# ---------------------------------------------------------------------------

def plot_objective_progress(monitor: HarmonyMonitor, run_dir: str, filename="objective_progress.png"):
    if not monitor.times:
        print("[WARN] No intermediate solutions recorded by HarmonyMonitor.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(monitor.times, monitor.objectives, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Objective value")
    plt.title("Objective progress during search")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(run_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved objective progress plot to: {out_path}")


# ---------------------------------------------------------------------------
# Main solve function
# ---------------------------------------------------------------------------

def solve_seating(excel_path: str, max_time_seconds: float = 60.0, 
                 base_run_dir: str = "runs", include_weak_preferences: bool = False):
    """Main entry point: solve, evaluate, and write a full run directory."""
    run_dir = create_run_directory(base_run_dir)
    
    mode_str = "WITH WEAK PREFERENCES" if include_weak_preferences else "STRONG PREFERENCES ONLY"
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION MODE: {mode_str}")
    print(f"{'='*70}\n")

    # 1) Load
    guests, adj_df, tbl_df, priority = load_inputs(excel_path)
    must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot, table_names = \
        build_preferences(guests, adj_df, tbl_df, include_weak=include_weak_preferences)
    chairs, chairs_by_table, round_tables, neighbor_pairs, chair_id_to_table = \
        build_chairs_and_neighbors(table_names)

    # 2) Build model
    model, x, adj_soft_pair_vars, guest_index = build_model(
        guests, priority, chairs, chairs_by_table, neighbor_pairs,
        table_names, must_pairs, cannot_pairs, adj_soft_scores, table_score, table_cannot
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
        for (cid, tname, pos) in chairs:
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
    print(f"[INFO] Saved seating plan to: {seating_path}")

    # 5) Build adjacency outcomes dataframe
    nbpairs = compute_neighbor_pairs_from_layout(seating_plan, chairs_by_table, round_tables)
    adj_set = compute_adjacency_set(seating_plan, nbpairs)

    def are_adjacent(g1, g2):
        return frozenset({g1, g2}) in adj_set

    adj_rows = []

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

    # soft adjacency pairs
    for (g1, g2, s) in [(g1, g2, s) for ((g1, g2), s) in adj_soft_scores.items()]:
        adj = are_adjacent(g1, g2)
        w = 0.5 * (priority[g1] + priority[g2])
        if s == 2:
            ptype = "Strongly prefer"
        elif s == -2:
            ptype = "Strongly avoid"
        elif s == 1:
            ptype = "Prefer"
        elif s == -1:
            ptype = "Avoid"
        else:
            ptype = "Unknown"
        
        adj_rows.append({
            "Guest1": g1,
            "Guest2": g2,
            "PreferenceType": ptype,
            "PreferenceScore": s,
            "PriorityAvg": w,
            "AreAdjacent": int(adj),
        })

    adjacency_outcomes = pd.DataFrame(adj_rows)
    adj_path = os.path.join(run_dir, "adjacency_outcomes.csv")
    adjacency_outcomes.to_csv(adj_path, index=False)
    print(f"[INFO] Saved adjacency outcomes to: {adj_path}")

    # 6) Table outcomes dataframe
    assigned_table = seating_plan.set_index("Guest")["Table"].to_dict()
    tbl_rows = []
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
    print(f"[INFO] Saved table outcomes to: {tbl_path}")

    # 7) Harmony metrics
    metrics = compute_harmony(
        guests, seating_plan, priority, adj_soft_scores, table_score, chairs_by_table, round_tables
    )
    metrics["raw_objective_value"] = best_objective
    metrics["include_weak_preferences"] = include_weak_preferences

    metrics_path = os.path.join(run_dir, "harmony_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Saved harmony metrics to: {metrics_path}")

    # 8) Compute satisfaction ratios
    ratios = compute_satisfaction_ratios(
        guests, seating_plan, must_pairs, cannot_pairs, adj_soft_scores,
        table_score, table_cannot, chairs_by_table, round_tables, priority, 
        include_weak=include_weak_preferences
    )
    
    ratios_path = os.path.join(run_dir, "satisfaction_ratios.json")
    with open(ratios_path, "w") as f:
        json.dump(ratios, f, indent=4)
    print(f"[INFO] Saved satisfaction ratios to: {ratios_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("CONSTRAINT SATISFACTION SUMMARY")
    print("="*60)
    print(f"Overall: {ratios['overall']['satisfied']}/{ratios['overall']['total']} "
          f"({ratios['overall']['ratio']*100:.1f}%) satisfied")
    print("\nAdjacency Constraints:")
    pref_types = ["must", "cannot", "strongly_prefer", "strongly_avoid"]
    if include_weak_preferences:
        pref_types.extend(["prefer", "avoid"])
    for pref_type in pref_types:
        data = ratios["adjacency"][pref_type]
        if data["total"] > 0:
            print(f"  {pref_type.replace('_', ' ').title()}: "
                  f"{data['satisfied']}/{data['total']} ({data['ratio']*100:.1f}%)")
    print("\nTable Constraints:")
    for pref_type in pref_types:
        data = ratios["table"][pref_type]
        if data["total"] > 0:
            print(f"  {pref_type.replace('_', ' ').title()}: "
                  f"{data['satisfied']}/{data['total']} ({data['ratio']*100:.1f}%)")
    print("="*60 + "\n")

    # 9) Metadata
    meta = {
        "run_dir": run_dir,
        "excel_path": excel_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "solver_status": solver.StatusName(status),
        "max_time_seconds": max_time_seconds,
        "num_guests": len(guests),
        "num_tables": len(table_names),
        "include_weak_preferences": include_weak_preferences,
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[INFO] Saved run metadata to: {meta_path}")

    # 10) Plots
    plot_objective_progress(monitor, run_dir)
    plot_satisfaction_ratios(ratios, run_dir, include_weak=include_weak_preferences)
    make_adjacency_result_heatmap(guests, adj_df, seating_plan, chairs_by_table, 
                                   round_tables, ratios, run_dir, 
                                   include_weak=include_weak_preferences)

    print(f"[DONE] Run complete. All outputs written to: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Family seating optimization solver with optional weak preferences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Strong preferences only (default):
  python %(prog)s preferences.xlsx --time 300
  
  # Include weak preferences (prefer/avoid):
  python %(prog)s preferences.xlsx --time 300 --include-weak-preferences
        """
    )
    parser.add_argument("excel_path", help="Path to Excel file with preferences.")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Maximum solve time in seconds (default: 60).")
    parser.add_argument("--runs-dir", type=str, default="runs",
                        help="Base directory for run outputs (default: 'runs').")
    parser.add_argument("--include-weak-preferences", action="store_true",
                        help="Include weak preferences (prefer/avoid) in optimization. "
                             "By default, only strong preferences are used.")

    args = parser.parse_args()
    solve_seating(args.excel_path, 
                 max_time_seconds=args.time, 
                 base_run_dir=args.runs_dir,
                 include_weak_preferences=args.include_weak_preferences)
