# Intelligent Workshop Spatial Tracking & Dynamic Layout Optimization System

Python + OpenCV + AI Layout Optimization software with an interactive **Streamlit** frontend for the RVCE Work Systems Design SEE project.

## Files
- `streamlit_app.py`: Interactive Streamlit frontend (Live Tool Finder, Spatial Heatmap, AI Affinity & ABC Analysis, Simulated Annealing Layout Optimizer, and sidebar event logger).
- `database.py`: SQLite schema (`workstations`, `tools`, `tool_events`) and spatial event logger.
- `cv_tracker.py`: Printable ArUco marker generator (`DICT_4X4_50`) and live USB webcam zone tracker.
- `ai_engine.py`: Co-usage affinity mining, ABC usage classification, and Simulated Annealing layout optimizer.
- `run_demo.py`: Generates printable ArUco markers, simulates 14 days of Formula Racing workshop logs, runs the AI optimizer, and saves `layout_optimization_results.png`.
- `dashboard.py`: Lightweight fallback local web dashboard (no Streamlit required).

## Quick Start (Windows / Mac / Linux)
```bash
pip install -r requirements.txt
python3 run_demo.py
streamlit run streamlit_app.py
```
To run live webcam tracking after printing stickers from `aruco_markers/`:
```bash
python3 -c "from cv_tracker import run_camera_tracker; run_camera_tracker(0)"
```
