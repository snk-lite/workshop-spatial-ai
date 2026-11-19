import random
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from database import seed_default_workshop, log_detection, DB_PATH
from cv_tracker import generate_aruco_markers
from ai_engine import (
    load_events_df,
    compute_co_usage_affinity,
    compute_bottlenecks_and_abc,
    optimize_layout_simulated_annealing,
)


def simulate_14_days(db_path: Path = DB_PATH, seed: int = 42) -> None:
    """Populate 14 days of realistic Formula Racing Club tool-usage events."""
    random.seed(seed)
    if db_path.exists():
        db_path.unlink()
    seed_default_workshop(db_path)

    zone_coords = {
        "Z_STORAGE": (1.0, 1.0),
        "Z_CHASSIS": (4.5, 1.5),
        "Z_SUSPENSION": (7.5, 2.0),
        "Z_ELECTRONICS": (6.0, 5.0),
    }
    start = datetime(2026, 11, 1, 9, 0, 0)

    for day in range(14):
        for hour_offset in range(0, 10, 2):
            ts = (start + timedelta(days=day, hours=hour_offset)).isoformat(timespec="seconds")
            # Chassis sessions: Torque Wrench + Socket Set + 12mm/13mm spanners co-used
            if random.random() < 0.85:
                for tid in ["TW-001", "SK-001", "SP-002", "SP-003"]:
                    if random.random() < 0.88:
                        xm, ym = zone_coords["Z_CHASSIS"]
                        log_detection(tid, "Z_CHASSIS", xm, ym, 0.96, ts, db_path)
            # Suspension sessions: 14mm/17mm spanners + Allen key set
            if random.random() < 0.75:
                for tid in ["SP-004", "SP-005", "AK-SET", "VC-001"]:
                    if random.random() < 0.80:
                        xm, ym = zone_coords["Z_SUSPENSION"]
                        log_detection(tid, "Z_SUSPENSION", xm, ym, 0.94, ts, db_path)
            # Electronics sessions: Multimeter + Wire Stripper + occasional 10mm spanner
            if random.random() < 0.70:
                for tid in ["MM-001", "WS-001", "SP-001"]:
                    if random.random() < 0.85:
                        xm, ym = zone_coords["Z_ELECTRONICS"]
                        log_detection(tid, "Z_ELECTRONICS", xm, ym, 0.95, ts, db_path)


def save_summary_chart(abc_df, opt_result: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Plot 1: Top Tools by Usage Events
    top = abc_df.head(8)
    axes[0].barh(top["tool_id"][::-1], top["usage_events"][::-1], color="#0f2942")
    axes[0].set_title("Tool Usage Frequency (14-Day Log)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Active Detections")

    # Plot 2: Before vs After Travel Distance
    labels = ["Baseline Layout", "AI-Optimized Layout"]
    vals = [opt_result["baseline_distance_m"], opt_result["optimized_distance_m"]]
    bars = axes[1].bar(labels, vals, color=["#c00000", "#047857"], width=0.5)
    axes[1].set_title(f"Total Retrieval Travel (-{opt_result['distance_reduction_pct']}%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Total Travel Distance (m)")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 5, f"{v} m", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180)
    plt.close()


if __name__ == "__main__":
    base = Path(__file__).parent
    simulate_14_days()
    markers = generate_aruco_markers(base / "aruco_markers")
    df = load_events_df()
    aff = compute_co_usage_affinity(df)
    abc = compute_bottlenecks_and_abc(df)
    opt = optimize_layout_simulated_annealing(df)
    save_summary_chart(abc, opt, base / "layout_optimization_results.png")

    print(f"Generated {len(markers)} printable ArUco markers in {base / 'aruco_markers'}")
    print(f"Logged {len(df)} spatial tool events across 14 simulated days.")
    print("\n--- Top Co-Used Tool Pairs (AI Affinity Mining) ---")
    print(aff.head(5).to_string(index=False))
    print("\n--- ABC & Bottleneck Summary ---")
    print(abc[["tool_id", "tool_name", "primary_usage_zone", "usage_events", "abc_class"]].to_string(index=False))
    print("\n--- Simulated Annealing Layout Optimization ---")
    print(f"Baseline Travel: {opt['baseline_distance_m']} m ({opt['baseline_retrieval_min']} min)")
    print(f"Optimized Travel: {opt['optimized_distance_m']} m ({opt['optimized_retrieval_min']} min)")
    print(f"Travel Reduction: {opt['distance_reduction_pct']}%")
    print("Recommended Relocations:")
    for r in opt["recommendations"]:
        print(f"  - {r['tool_id']} ({r['tool_name']}): {r['current_zone']} -> {r['recommended_zone']}")
