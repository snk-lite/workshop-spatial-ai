import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from database import get_connection, DB_PATH


def load_events_df(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = pd.read_sql_query("""
        SELECT e.event_id, e.tool_id, t.tool_name, t.category, t.designated_zone,
               e.zone_id, w.zone_name, e.x_m, e.y_m, e.timestamp, e.confidence, e.is_misplaced
        FROM tool_events e
        JOIN tools t ON e.tool_id = t.tool_id
        JOIN workstations w ON e.zone_id = w.zone_id
        ORDER BY e.timestamp ASC
    """, conn)
    conn.close()
    return df


def compute_co_usage_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """
    AI Pattern Mining (Association / Co-Occurrence Analysis):
    Finds tools frequently used together in the same workstation session (1-hour windows).
    """
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["session"] = pd.to_datetime(work["timestamp"]).dt.floor("1h").astype(str) + "_" + work["zone_id"]
    sessions = work.groupby("session")["tool_id"].apply(lambda s: sorted(set(s))).tolist()

    pair_counts = {}
    tool_counts = {}
    for tools in sessions:
        for t in tools:
            tool_counts[t] = tool_counts.get(t, 0) + 1
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                pair = (tools[i], tools[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rows = []
    n_sessions = max(1, len(sessions))
    for (t1, t2), cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        support = cnt / n_sessions
        conf = cnt / max(1, min(tool_counts[t1], tool_counts[t2]))
        if cnt >= 3:
            rows.append({
                "tool_a": t1,
                "tool_b": t2,
                "co_occurrences": cnt,
                "support": round(support, 3),
                "confidence": round(conf, 3),
            })
    return pd.DataFrame(rows)


def compute_bottlenecks_and_abc(df: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    AI Usage Clustering & ABC Analysis:
    Scores each tool's total detections, primary usage zone, misplacement rate, and sharing conflicts.
    """
    conn = get_connection(db_path)
    tools_df = pd.read_sql_query("SELECT tool_id, tool_name, designated_zone FROM tools", conn)
    zones_df = pd.read_sql_query("SELECT zone_id, center_x_m, center_y_m FROM workstations", conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    zone_coords = zones_df.set_index("zone_id")[["center_x_m", "center_y_m"]].to_dict("index")

    stats = []
    for _, t in tools_df.iterrows():
        tid = t["tool_id"]
        sub = df[df["tool_id"] == tid]
        if sub.empty:
            continue
        active_sub = sub[sub["zone_id"] != "Z_STORAGE"]
        primary_zone = active_sub["zone_id"].mode()[0] if not active_sub.empty else t["designated_zone"]
        unique_zones = active_sub["zone_id"].nunique()
        misplaced_rate = float(sub["is_misplaced"].mean())

        # Travel distance from designated_zone to each usage event
        dz = zone_coords[t["designated_zone"]]
        total_dist = 0.0
        for _, r in active_sub.iterrows():
            uz = zone_coords[r["zone_id"]]
            total_dist += math.hypot(uz["center_x_m"] - dz["center_x_m"], uz["center_y_m"] - dz["center_y_m"])

        stats.append({
            "tool_id": tid,
            "tool_name": t["tool_name"],
            "designated_zone": t["designated_zone"],
            "primary_usage_zone": primary_zone,
            "usage_events": int(len(active_sub)),
            "workstations_sharing": int(unique_zones),
            "misplaced_pct": round(misplaced_rate * 100, 1),
            "current_travel_m": round(total_dist, 1),
        })

    res = pd.DataFrame(stats).sort_values("usage_events", ascending=False).reset_index(drop=True)
    # ABC class by cumulative usage
    cum = res["usage_events"].cumsum() / max(1, res["usage_events"].sum())
    res["abc_class"] = np.where(cum <= 0.65, "A (High)", np.where(cum <= 0.90, "B (Medium)", "C (Low)"))
    return res


def optimize_layout_simulated_annealing(df: pd.DataFrame, db_path: Path = DB_PATH, seed: int = 42) -> dict:
    """
    Simulated Annealing Facility Layout Optimizer:
    Minimizes Z = w1*RetrievalTime + w2*TravelDistance + w3*SharingConflicts + w4*SpacePenalty
    to recommend optimal storage zone assignments for each tool.
    """
    random.seed(seed)
    conn = get_connection(db_path)
    tools_df = pd.read_sql_query("SELECT tool_id, tool_name, designated_zone FROM tools", conn)
    zones_df = pd.read_sql_query("SELECT zone_id, zone_name, center_x_m, center_y_m FROM workstations", conn)
    conn.close()

    zone_ids = zones_df["zone_id"].tolist()
    zone_names = zones_df.set_index("zone_id")["zone_name"].to_dict()
    zone_coords = zones_df.set_index("zone_id")[["center_x_m", "center_y_m"]].to_dict("index")

    # Demand matrix: tool_id -> {zone_id: count}
    active = df[df["zone_id"] != "Z_STORAGE"]
    demand = {}
    for tid in tools_df["tool_id"]:
        sub = active[active["tool_id"] == tid]
        demand[tid] = sub["zone_id"].value_counts().to_dict()

    # Co-usage pairs for affinity bonus
    aff_df = compute_co_usage_affinity(df)
    aff_pairs = [(r["tool_a"], r["tool_b"], r["co_occurrences"]) for _, r in aff_df.iterrows()]

    def objective(assignment: dict) -> tuple[float, float, float]:
        total_dist = 0.0
        zone_load = {z: 0 for z in zone_ids}
        for tid, home_z in assignment.items():
            zone_load[home_z] += 1
            hc = zone_coords[home_z]
            for use_z, cnt in demand.get(tid, {}).items():
                uc = zone_coords[use_z]
                d = math.hypot(uc["center_x_m"] - hc["center_x_m"], uc["center_y_m"] - hc["center_y_m"])
                total_dist += d * cnt

        # Affinity penalty if frequently co-used tools are stored in different zones
        aff_penalty = 0.0
        for ta, tb, cnt in aff_pairs:
            if assignment.get(ta) != assignment.get(tb):
                aff_penalty += cnt * 0.8

        # Space capacity penalty (prefer <= 4 tools per bench zone)
        cap_penalty = sum(max(0, c - 4) * 15.0 for z, c in zone_load.items() if z != "Z_STORAGE")
        # Assume walking speed 1.2 m/s + 15s search overhead when far
        est_retrieval_s = total_dist / 1.2
        z_score = 1.0 * total_dist + 0.5 * est_retrieval_s + 0.4 * aff_penalty + cap_penalty
        return z_score, total_dist, est_retrieval_s

    baseline_assign = tools_df.set_index("tool_id")["designated_zone"].to_dict()
    base_z, base_dist, base_time = objective(baseline_assign)

    cur_assign = baseline_assign.copy()
    cur_z = base_z
    best_assign = cur_assign.copy()
    best_z, best_dist, best_time = base_z, base_dist, base_time

    temp = 100.0
    cooling = 0.992
    tool_list = list(cur_assign.keys())
    for _ in range(1500):
        cand = cur_assign.copy()
        t_pick = random.choice(tool_list)
        cand[t_pick] = random.choice(zone_ids)
        cand_z, cand_dist, cand_time = objective(cand)
        delta = cand_z - cur_z
        if delta < 0 or random.random() < math.exp(-delta / max(0.01, temp)):
            cur_assign = cand
            cur_z = cand_z
            if cur_z < best_z:
                best_assign = cand.copy()
                best_z, best_dist, best_time = cand_z, cand_dist, cand_time
        temp *= cooling

    recommendations = []
    for _, row in tools_df.iterrows():
        tid = row["tool_id"]
        old_z = baseline_assign[tid]
        new_z = best_assign[tid]
        if old_z != new_z:
            recommendations.append({
                "tool_id": tid,
                "tool_name": row["tool_name"],
                "current_zone": zone_names[old_z],
                "recommended_zone": zone_names[new_z],
                "reason": "High local demand & co-usage affinity"
            })

    dist_red = ((base_dist - best_dist) / max(1e-6, base_dist)) * 100.0
    return {
        "baseline_distance_m": round(base_dist, 1),
        "optimized_distance_m": round(best_dist, 1),
        "distance_reduction_pct": round(dist_red, 1),
        "baseline_retrieval_min": round(base_time / 60.0, 1),
        "optimized_retrieval_min": round(best_time / 60.0, 1),
        "recommendations": recommendations,
    }
