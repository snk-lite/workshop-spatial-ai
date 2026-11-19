import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from database import get_connection
from ai_engine import load_events_df, compute_co_usage_affinity, compute_bottlenecks_and_abc, optimize_layout_simulated_annealing

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RVCE Workshop Spatial Intelligence Dashboard</title>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f1f5f9; color: #0f172a; }
  header { background: #0f2942; color: white; padding: 18px 32px; display: flex; justify-content: space-between; align-items: center; border-bottom: 4px solid #c00000; }
  h1 { margin: 0; font-size: 20px; }
  .sub { font-size: 13px; color: #cbd5e1; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; padding: 24px 32px 0; }
  .card { background: white; border-radius: 12px; padding: 18px; box-shadow: 0 2px 8px rgba(15,23,42,0.06); }
  .kpi { font-size: 28px; font-weight: 700; color: #0f2942; }
  .kpi.red { color: #c00000; }
  .kpi.green { color: #047857; }
  .section { padding: 20px 32px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(15,23,42,0.06); }
  th { background: #0f2942; color: white; text-align: left; padding: 10px 14px; font-size: 13px; }
  td { padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
  .badge { padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .ok { background: #d1fae5; color: #065f46; }
  .warn { background: #fee2e2; color: #991b1b; }
</style>
</head>
<body>
<header>
  <div>
    <h1>RVCE Workshop Spatial Intelligence & Layout Optimizer</h1>
    <div class="sub">Formula Racing Club Live Spatial Inventory & AI Recommendations</div>
  </div>
  <div class="sub">Work Systems Design — SEE Project</div>
</header>
<div class="grid">
  <div class="card"><div class="sub">Baseline Travel Distance</div><div class="kpi">__BASE_DIST__ m</div></div>
  <div class="card"><div class="sub">Optimized Travel Distance</div><div class="kpi green">__OPT_DIST__ m</div></div>
  <div class="card"><div class="sub">Travel Reduction</div><div class="kpi red">-__DIST_RED__%</div></div>
  <div class="card"><div class="sub">Retrieval Time Saved</div><div class="kpi green">__BASE_TIME__ → __OPT_TIME__ min</div></div>
</div>
<div class="section">
  <h3>Live Tool Inventory & Last-Detected Location</h3>
  <table>
    <thead><tr><th>Tool ID</th><th>Name</th><th>Designated Zone</th><th>Last Seen Zone</th><th>Last Seen Time</th><th>Status</th></tr></thead>
    <tbody>__TOOL_ROWS__</tbody>
  </table>
</div>
<div class="section">
  <h3>AI Layout Redesign Recommendations (Simulated Annealing)</h3>
  <table>
    <thead><tr><th>Tool ID</th><th>Name</th><th>Current Storage</th><th>Recommended Storage</th><th>Rationale</th></tr></thead>
    <tbody>__REC_ROWS__</tbody>
  </table>
</div>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        conn = get_connection()
        tools = conn.execute("""
            SELECT t.tool_id, t.tool_name, w1.zone_name AS des_name, w2.zone_name AS last_name, t.last_seen_ts,
                   CASE WHEN t.designated_zone != t.last_detected_zone AND t.last_detected_zone != 'Z_STORAGE' THEN 'Misplaced' ELSE 'In Place' END AS status
            FROM tools t
            LEFT JOIN workstations w1 ON t.designated_zone = w1.zone_id
            LEFT JOIN workstations w2 ON t.last_detected_zone = w2.zone_id
        """).fetchall()
        conn.close()

        df = load_events_df()
        opt = optimize_layout_simulated_annealing(df)

        tool_rows = "".join(
            f"<tr><td><b>{r['tool_id']}</b></td><td>{r['tool_name']}</td><td>{r['des_name']}</td>"
            f"<td>{r['last_name']}</td><td>{r['last_seen_ts']}</td>"
            f"<td><span class='badge {'warn' if r['status']=='Misplaced' else 'ok'}'>{r['status']}</span></td></tr>"
            for r in tools
        )
        rec_rows = "".join(
            f"<tr><td><b>{r['tool_id']}</b></td><td>{r['tool_name']}</td><td>{r['current_zone']}</td>"
            f"<td><b>{r['recommended_zone']}</b></td><td>{r['reason']}</td></tr>"
            for r in opt["recommendations"]
        )

        html = (HTML_TEMPLATE
                .replace("__BASE_DIST__", str(opt["baseline_distance_m"]))
                .replace("__OPT_DIST__", str(opt["optimized_distance_m"]))
                .replace("__DIST_RED__", str(opt["distance_reduction_pct"]))
                .replace("__BASE_TIME__", str(opt["baseline_retrieval_min"]))
                .replace("__OPT_TIME__", str(opt["optimized_retrieval_min"]))
                .replace("__TOOL_ROWS__", tool_rows)
                .replace("__REC_ROWS__", rec_rows))

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8050), DashboardHandler)
    print("Dashboard running at http://127.0.0.1:8050 (Ctrl+C to stop)")
    server.serve_forever()
