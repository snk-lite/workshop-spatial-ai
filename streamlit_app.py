import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from database import get_connection, seed_default_workshop, log_detection, DB_PATH
import tempfile
from cv_tracker import generate_aruco_markers, process_video_file
from ai_engine import (
    load_events_df,
    compute_co_usage_affinity,
    compute_bottlenecks_and_abc,
    optimize_layout_simulated_annealing,
)
from run_demo import simulate_14_days

st.set_page_config(
    page_title="RVCE Workshop Spatial Intelligence",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean executive cards & RVCE navy/crimson accents
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-banner {
        background: linear-gradient(135deg, #0F2942 0%, #1E3A5F 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        border-left: 6px solid #C00000;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero-banner h1 { margin: 0; font-size: 1.65rem; color: white; }
    .hero-banner p { margin: 0.35rem 0 0; color: #CBD5E1; font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 14px 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Ensure DB exists on first launch
if not DB_PATH.exists():
    simulate_14_days(DB_PATH)

# ── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
st.sidebar.title("Control Panel")
st.sidebar.caption("RVCE Formula Racing & Aerospace Workshop")

if st.sidebar.button("Re-run 14-Day Simulation", use_container_width=True):
    simulate_14_days(DB_PATH)
    st.sidebar.success("Simulated 14 days of workshop logs!")

if st.sidebar.button("Generate Printable ArUco Markers", use_container_width=True):
    markers = generate_aruco_markers(Path(__file__).parent / "aruco_markers")
    st.sidebar.success(f"Generated {len(markers)} markers in aruco_markers/")

st.sidebar.divider()
st.sidebar.subheader("Optimizer Parameters")
sim_seed = st.sidebar.slider("Simulated Annealing Seed", min_value=1, max_value=100, value=42)
zone_filter = st.sidebar.multiselect(
    "Filter by Workstation Zone",
    options=["Z_STORAGE", "Z_CHASSIS", "Z_SUSPENSION", "Z_ELECTRONICS"],
    default=["Z_STORAGE", "Z_CHASSIS", "Z_SUSPENSION", "Z_ELECTRONICS"],
)

st.sidebar.divider()
st.sidebar.subheader("Quick Log Tool Movement")
conn = get_connection(DB_PATH)
all_tools = pd.read_sql_query("SELECT tool_id, tool_name FROM tools ORDER BY tool_id", conn)
all_zones = pd.read_sql_query("SELECT zone_id, zone_name, center_x_m, center_y_m FROM workstations", conn)
conn.close()

pick_tool = st.sidebar.selectbox("Tool", all_tools["tool_id"] + " — " + all_tools["tool_name"])
pick_zone = st.sidebar.selectbox("Detected at Zone", all_zones["zone_id"] + " — " + all_zones["zone_name"])
if st.sidebar.button("Log Detection Event", type="primary", use_container_width=True):
    tid = pick_tool.split(" — ")[0]
    zid = pick_zone.split(" — ")[0]
    zrow = all_zones[all_zones["zone_id"] == zid].iloc[0]
    log_detection(tid, zid, float(zrow["center_x_m"]), float(zrow["center_y_m"]), confidence=0.97, db_path=DB_PATH)
    st.sidebar.success(f"Logged {tid} at {zid}")

# ── LOAD DATA & RUN AI ENGINE ────────────────────────────────────────────────
df = load_events_df(DB_PATH)
if zone_filter:
    df_filtered = df[df["zone_id"].isin(zone_filter)]
else:
    df_filtered = df

aff_df = compute_co_usage_affinity(df)
abc_df = compute_bottlenecks_and_abc(df, DB_PATH)
opt = optimize_layout_simulated_annealing(df, DB_PATH, seed=sim_seed)

conn = get_connection(DB_PATH)
tools_live = pd.read_sql_query("""
    SELECT t.tool_id, t.tool_name, t.category, w1.zone_name AS designated_storage,
           w2.zone_name AS last_seen_zone, t.last_seen_ts, t.visibility_status, t.confidence,
           CASE WHEN t.designated_zone != t.last_detected_zone AND t.last_detected_zone != 'Z_STORAGE'
                THEN 'Misplaced' ELSE 'In Place' END AS status
    FROM tools t
    LEFT JOIN workstations w1 ON t.designated_zone = w1.zone_id
    LEFT JOIN workstations w2 ON t.last_detected_zone = w2.zone_id
    ORDER BY t.tool_id
""", conn)
conn.close()

misplaced_count = int((tools_live["status"] == "Misplaced").sum())

# ── HEADER & KPI ROW ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>Intelligent Workshop Spatial Tracking & Dynamic Layout Optimizer</h1>
  <p>Work Systems Design (SEE Project) · Live Spatial Inventory, AI Co-Usage Mining & Simulated Annealing Layout Engine</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Tracked Tools", f"{len(tools_live)}", f"{misplaced_count} currently away from storage", delta_color="inverse")
k2.metric("Baseline Travel Distance", f"{opt['baseline_distance_m']} m", f"{opt['baseline_retrieval_min']} min retrieval")
k3.metric("AI-Optimized Travel", f"{opt['optimized_distance_m']} m", f"-{opt['distance_reduction_pct']}% reduction")
k4.metric("Logged Spatial Events", f"{len(df)}", "14-day + live detections")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Live Tool Finder",
    "🗺️ Spatial Heatmap & Zones",
    "🧠 AI Affinity & Bottlenecks",
    "⚙️ Layout Optimizer",
    "🎥 Video Tracker & Demo",
])

with tab1:
    st.subheader("Where Is Every Tool Right Now?")
    search_q = st.text_input("Search tool by ID, name, or category (e.g. 'Spanner', 'TW-001')", "")
    view_df = tools_live.copy()
    if search_q.strip():
        mask = (
            view_df["tool_id"].str.contains(search_q, case=False, na=False)
            | view_df["tool_name"].str.contains(search_q, case=False, na=False)
            | view_df["category"].str.contains(search_q, case=False, na=False)
        )
        view_df = view_df[mask]

    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

with tab2:
    st.subheader("Workshop Zone Activity & Tool Accumulation")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        zone_counts = df_filtered.groupby("zone_name").size().reset_index(name="detections")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#0F2942", "#C00000", "#1E3A5F", "#047857"][: len(zone_counts)]
        ax.barh(zone_counts["zone_name"], zone_counts["detections"], color=colors)
        ax.set_xlabel("Total Tool Detections")
        ax.set_title("Tool Activity by Workstation Zone", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.markdown("**Zone Coordinate Map (Meters)**")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        for _, zr in all_zones.iterrows():
            cnt = int((df_filtered["zone_id"] == zr["zone_id"]).sum())
            ax2.scatter(zr["center_x_m"], zr["center_y_m"], s=max(200, cnt * 6), alpha=0.75, color="#0F2942" if zr["zone_id"] != "Z_CHASSIS" else "#C00000")
            ax2.text(zr["center_x_m"], zr["center_y_m"] + 0.35, f"{zr['zone_name'].split(' ')[0]}\n({cnt} events)", ha="center", fontsize=9, fontweight="bold")
        ax2.set_xlim(0, 9)
        ax2.set_ylim(0, 6.5)
        ax2.set_xlabel("Workshop X (meters)")
        ax2.set_ylabel("Workshop Y (meters)")
        ax2.set_title("Spatial Distribution Bubble Map", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2, use_container_width=True)

with tab3:
    st.subheader("AI Pattern Mining: Co-Used Tool Pairs & ABC Analysis")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Frequently Co-Used Tool Pairs (Association Rules)**")
        if not aff_df.empty:
            st.dataframe(aff_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough co-occurrence sessions yet.")
    with col_b:
        st.markdown("**ABC Usage Classification & Current Travel**")
        if not abc_df.empty:
            st.dataframe(
                abc_df[["tool_id", "tool_name", "primary_usage_zone", "usage_events", "current_travel_m", "abc_class"]],
                use_container_width=True,
                hide_index=True,
            )

with tab4:
    st.subheader("Simulated Annealing Layout Recommendations")
    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        rec_df = pd.DataFrame(opt["recommendations"])
        if not rec_df.empty:
            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.success("Current layout is already optimal for observed demand!")
    with col_r:
        fig3, ax3 = plt.subplots(figsize=(5, 3.8))
        bars = ax3.bar(["Baseline\nLayout", "AI-Optimized\nLayout"], [opt["baseline_distance_m"], opt["optimized_distance_m"]], color=["#C00000", "#047857"], width=0.5)
        ax3.set_ylabel("Total Retrieval Distance (m)")
        ax3.set_title(f"Before vs. After (-{opt['distance_reduction_pct']}%)", fontweight="bold")
        for b, v in zip(bars, [opt["baseline_distance_m"], opt["optimized_distance_m"]]):
            ax3.text(b.get_x() + b.get_width() / 2, v + 20, f"{v} m", ha="center", fontweight="bold")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        st.pyplot(fig3, use_container_width=True)

with tab5:
    st.subheader("Upload Workshop Video (.mp4 / .mov / .avi)")
    st.caption("Uses ArUco marker tracking when stickers are present, or Motion-Zone Demo Mode for stock workshop clips.")
    track_mode = st.radio(
        "Tracking Mode",
        options=["auto", "aruco_only"],
        format_func=lambda x: "Auto (ArUco + Motion-Zone Demo Fallback)" if x == "auto" else "ArUco Markers Only (Tagged Tools)",
        horizontal=True,
    )
    uploaded_vid = st.file_uploader("Upload 1-minute workshop footage", type=["mp4", "mov", "avi"])
    if uploaded_vid is not None:
        if st.button("Analyze Video & Log Detections", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name
            with st.spinner("Running OpenCV zone & marker detection..."):
                res = process_video_file(tmp_path, mode=track_mode, db_path=DB_PATH)
            st.success(f"Processed {res['frames_processed']} frames — ArUco hits: {res['aruco_detections']}, Motion-Zone hits: {res['motion_zone_detections']}")
            if res["preview_frames"]:
                cols = st.columns(3)
                for idx, frm in enumerate(res["preview_frames"][:6]):
                    cols[idx % 3].image(frm, caption=f"Sampled Frame #{idx+1}", use_container_width=True)
            st.info("Detections have been logged to SQLite! Switch tabs or refresh to see updated KPIs.")
