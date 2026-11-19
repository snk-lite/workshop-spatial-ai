import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "workshop.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS workstations (
        zone_id TEXT PRIMARY KEY,
        zone_name TEXT NOT NULL,
        x_min REAL NOT NULL,
        y_min REAL NOT NULL,
        x_max REAL NOT NULL,
        y_max REAL NOT NULL,
        center_x_m REAL NOT NULL,
        center_y_m REAL NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tools (
        tool_id TEXT PRIMARY KEY,
        aruco_id INTEGER UNIQUE NOT NULL,
        tool_name TEXT NOT NULL,
        category TEXT NOT NULL,
        designated_zone TEXT NOT NULL,
        last_detected_zone TEXT,
        last_seen_ts TEXT,
        visibility_status TEXT DEFAULT 'Unknown',
        confidence REAL DEFAULT 0.0,
        FOREIGN KEY (designated_zone) REFERENCES workstations(zone_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tool_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_id TEXT NOT NULL,
        zone_id TEXT NOT NULL,
        x_m REAL NOT NULL,
        y_m REAL NOT NULL,
        timestamp TEXT NOT NULL,
        confidence REAL NOT NULL,
        is_misplaced INTEGER DEFAULT 0,
        FOREIGN KEY (tool_id) REFERENCES tools(tool_id),
        FOREIGN KEY (zone_id) REFERENCES workstations(zone_id)
    )
    """)

    conn.commit()
    conn.close()


def seed_default_workshop(db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    zones = [
        ("Z_STORAGE", "Central Tool Board", 0.0, 0.0, 0.30, 0.40, 1.0, 1.0),
        ("Z_CHASSIS", "Chassis Fabrication Bench", 0.30, 0.0, 0.65, 0.50, 4.5, 1.5),
        ("Z_SUSPENSION", "Suspension & Drivetrain Bench", 0.65, 0.0, 1.0, 0.50, 7.5, 2.0),
        ("Z_ELECTRONICS", "Electronics & Battery Bench", 0.30, 0.50, 1.0, 1.0, 6.0, 5.0),
    ]
    cur.executemany("""
        INSERT OR REPLACE INTO workstations
        (zone_id, zone_name, x_min, y_min, x_max, y_max, center_x_m, center_y_m)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, zones)

    tools = [
        ("SP-001", 1, "10mm Spanner", "Spanner", "Z_STORAGE"),
        ("SP-002", 2, "12mm Spanner", "Spanner", "Z_STORAGE"),
        ("SP-003", 3, "13mm Spanner", "Spanner", "Z_STORAGE"),
        ("SP-004", 4, "14mm Spanner", "Spanner", "Z_STORAGE"),
        ("SP-005", 5, "17mm Spanner", "Spanner", "Z_STORAGE"),
        ("TW-001", 6, "Torque Wrench (1/2 in)", "Torque", "Z_STORAGE"),
        ("SK-001", 7, "Socket Set (10-19mm)", "Socket", "Z_STORAGE"),
        ("AK-SET", 8, "Allen Key Holder (2-10mm)", "Allen Key", "Z_STORAGE"),
        ("VC-001", 9, "Vernier Caliper", "Measuring", "Z_STORAGE"),
        ("MM-001", 10, "Digital Multimeter", "Electronics", "Z_STORAGE"),
        ("WS-001", 11, "Wire Stripper & Crimper", "Electronics", "Z_STORAGE"),
        ("AG-001", 12, "Angle Grinder", "Power Tool", "Z_STORAGE"),
    ]
    for tid, aid, tname, cat, dzone in tools:
        cur.execute("""
            INSERT OR IGNORE INTO tools
            (tool_id, aruco_id, tool_name, category, designated_zone, last_detected_zone, last_seen_ts, visibility_status, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Visible', 1.0)
        """, (tid, aid, tname, cat, dzone, dzone, datetime.now().isoformat(timespec="seconds")))

    conn.commit()
    conn.close()


def log_detection(tool_id: str, zone_id: str, x_m: float, y_m: float, confidence: float, ts: str = None, db_path: Path = DB_PATH) -> None:
    ts = ts or datetime.now().isoformat(timespec="seconds")
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT designated_zone FROM tools WHERE tool_id = ?", (tool_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    is_misplaced = 1 if (row["designated_zone"] != zone_id and zone_id != "Z_STORAGE") else 0

    cur.execute("""
        INSERT INTO tool_events (tool_id, zone_id, x_m, y_m, timestamp, confidence, is_misplaced)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (tool_id, zone_id, x_m, y_m, ts, confidence, is_misplaced))

    cur.execute("""
        UPDATE tools
        SET last_detected_zone = ?, last_seen_ts = ?, visibility_status = 'Visible', confidence = ?
        WHERE tool_id = ?
    """, (zone_id, ts, confidence, tool_id))

    conn.commit()
    conn.close()
