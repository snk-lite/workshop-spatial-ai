import cv2
import numpy as np
from pathlib import Path
from database import get_connection, log_detection, DB_PATH

ARUCO_DICT = cv2.aruco.DICT_4X4_50


def generate_aruco_markers(out_dir: Path, marker_size_px: int = 220) -> list[Path]:
    """Generate printable ArUco marker PNGs for every tool in the database."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    conn = get_connection()
    tools = conn.execute("SELECT tool_id, aruco_id, tool_name FROM tools ORDER BY aruco_id").fetchall()
    conn.close()

    generated = []
    for t in tools:
        img = cv2.aruco.generateImageMarker(aruco_dict, int(t["aruco_id"]), marker_size_px)
        canvas = np.full((marker_size_px + 60, marker_size_px + 40), 255, dtype=np.uint8)
        canvas[20:20 + marker_size_px, 20:20 + marker_size_px] = img
        cv2.putText(
            canvas, f"{t['tool_id']} (ID:{t['aruco_id']})",
            (18, marker_size_px + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,), 2
        )
        p = out_dir / f"{t['tool_id']}_aruco_{t['aruco_id']}.png"
        cv2.imwrite(str(p), canvas)
        generated.append(p)
    return generated


def map_point_to_zone(norm_x: float, norm_y: float, zones: list) -> tuple[str, float, float]:
    """Map normalized image coordinates [0..1] to calibrated workstation zone and meters."""
    for z in zones:
        if z["x_min"] <= norm_x <= z["x_max"] and z["y_min"] <= norm_y <= z["y_max"]:
            return z["zone_id"], z["center_x_m"], z["center_y_m"]
    return "Z_STORAGE", 1.0, 1.0


def process_video_file(video_path: str, mode: str = "auto", db_path: Path = DB_PATH, sample_stride: int = 10, max_sampled: int = 24) -> dict:
    """
    Process a recorded .mp4/.mov/.avi workshop video.
    - Detects ArUco markers when present.
    - In 'auto' or 'demo_motion' mode, falls back to OpenCV motion-contour detection
      when stock videos lack ArUco stickers, mapping active regions to workshop zones.
    Returns annotated RGB preview frames and detection summary.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    conn = get_connection(db_path)
    zones = conn.execute("SELECT * FROM workstations").fetchall()
    tool_rows = conn.execute("SELECT tool_id, aruco_id FROM tools ORDER BY aruco_id").fetchall()
    tool_map = {row["aruco_id"]: row["tool_id"] for row in tool_rows}
    zone_demo_tools = {
        "Z_STORAGE": "SP-001",
        "Z_CHASSIS": "TW-001",
        "Z_SUSPENSION": "AK-SET",
        "Z_ELECTRONICS": "MM-001",
    }
    conn.close()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    prev_gray = None
    frame_idx = 0
    sampled_frames = []
    aruco_hits = 0
    motion_hits = 0
    zone_activity = {}

    while len(sampled_frames) < max_sampled:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_idx % sample_stride != 1:
            prev_gray = cv2.GaussianBlur(gray, (11, 11), 0)
            continue

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Draw workstation zone overlays
        for z in zones:
            pt1 = (int(z["x_min"] * w), int(z["y_min"] * h))
            pt2 = (int(z["x_max"] * w), int(z["y_max"] * h))
            cv2.rectangle(annotated, pt1, pt2, (66, 41, 15), 2)
            cv2.putText(annotated, z["zone_name"].split(" ")[0], (pt1[0] + 8, pt1[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (66, 41, 15), 2)

        # 1. Try ArUco detection
        corners, ids, _ = detector.detectMarkers(gray)
        found_aruco = False
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                if int(mid) in tool_map:
                    found_aruco = True
                    aruco_hits += 1
                    pts = c[0]
                    cx = float(np.mean(pts[:, 0])) / w
                    cy = float(np.mean(pts[:, 1])) / h
                    zone_id, xm, ym = map_point_to_zone(cx, cy, zones)
                    log_detection(tool_map[int(mid)], zone_id, xm, ym, confidence=0.96, db_path=db_path)
                    zone_activity[zone_id] = zone_activity.get(zone_id, 0) + 1

        # 2. Motion-zone fallback for stock videos without ArUco stickers
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        if not found_aruco and mode in ("auto", "demo_motion") and prev_gray is not None:
            diff = cv2.absdiff(prev_gray, blur)
            _, thresh = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (w * h) * 0.004
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(cnt) < min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = (x + bw / 2.0) / w
                cy = (y + bh / 2.0) / h
                zone_id, xm, ym = map_point_to_zone(cx, cy, zones)
                demo_tid = zone_demo_tools.get(zone_id, "SP-002")
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 192), 2)
                cv2.putText(annotated, f"Active: {demo_tid} ({zone_id})", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 192), 2)
                log_detection(demo_tid, zone_id, xm, ym, confidence=0.82, db_path=db_path)
                motion_hits += 1
                zone_activity[zone_id] = zone_activity.get(zone_id, 0) + 1

        prev_gray = blur
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        sampled_frames.append(rgb)

    cap.release()
    return {
        "frames_processed": frame_idx,
        "preview_frames": sampled_frames,
        "aruco_detections": aruco_hits,
        "motion_zone_detections": motion_hits,
        "zone_activity": zone_activity,
    }


def run_camera_tracker(source=0, db_path: Path = DB_PATH, max_frames: int = 0) -> None:
    """
    Run live OpenCV ArUco tracking on a USB webcam (source=0) or video file.
    Press 'q' to quit the live preview window.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    conn = get_connection(db_path)
    zones = conn.execute("SELECT * FROM workstations").fetchall()
    tool_map = {row["aruco_id"]: row["tool_id"] for row in conn.execute("SELECT tool_id, aruco_id FROM tools").fetchall()}
    conn.close()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {source}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        for z in zones:
            pt1 = (int(z["x_min"] * w), int(z["y_min"] * h))
            pt2 = (int(z["x_max"] * w), int(z["y_max"] * h))
            cv2.rectangle(frame, pt1, pt2, (15, 41, 66), 2)
            cv2.putText(frame, z["zone_name"], (pt1[0] + 8, pt1[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 41, 66), 2)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                if int(mid) in tool_map and frame_count % 15 == 0:
                    pts = c[0]
                    cx = float(np.mean(pts[:, 0])) / w
                    cy = float(np.mean(pts[:, 1])) / h
                    zone_id, xm, ym = map_point_to_zone(cx, cy, zones)
                    log_detection(tool_map[int(mid)], zone_id, xm, ym, confidence=0.95, db_path=db_path)

        cv2.imshow("RVCE Workshop Spatial Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
