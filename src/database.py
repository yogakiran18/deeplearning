"""Road Damage Detection Relational Database (SQLite) Module.

This module provides persistent relational storage for road damage inspections,
individual defect detections (D00, D10, D20, D40), PCI condition assessments,
and model metadata.

Why this module exists:
- Eliminates reliance on isolated CSV logs by providing indexed relational querying.
- Preserves complete spatial defect bounding boxes, confidence values, and timestamps.
- Enables historical roadway degradation tracking, maintenance scheduling, and GIS export.
- Seamlessly integrates with CLI inference scripts and the Flask inspection dashboard.

Schema Overview:
- `inspections`: Metadata for each processed image, video, or stream (PCI score, counts, file paths).
- `detections`: Granular defect records linked to an inspection (class, box coords, confidence).
- `models`: Checkpoint registry tracking weights, versions, and validation metrics.
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Default database location within git-ignored outputs directory
DEFAULT_DB_PATH = Path("outputs/road_damage.db")

# Standard RDD severity weights for PCI calculation
SEVERITY_WEIGHTS: Dict[str, int] = {
    "D00": 2,   # Longitudinal Crack
    "D10": 2,   # Transverse Crack
    "D20": 6,   # Alligator Crack
    "D40": 12,  # Pothole
}

CLASS_NAMES: Dict[str, str] = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite database connection with row dictionary access."""
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> Path:
    """Initialize SQLite database tables, schema, and performance indexes."""
    path = db_path or DEFAULT_DB_PATH
    with get_connection(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                media_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                duration_sec REAL DEFAULT 0.0,
                total_frames INTEGER DEFAULT 1,
                processed_frames INTEGER DEFAULT 1,
                fps REAL DEFAULT 0.0,
                pci_score REAL DEFAULT 100.0,
                pci_rating TEXT DEFAULT 'Good',
                total_defects INTEGER DEFAULT 0,
                d00_count INTEGER DEFAULT 0,
                d10_count INTEGER DEFAULT 0,
                d20_count INTEGER DEFAULT 0,
                d40_count INTEGER DEFAULT 0,
                media_path TEXT,
                output_path TEXT,
                csv_path TEXT,
                model_used TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id TEXT NOT NULL,
                frame_index INTEGER DEFAULT 0,
                timestamp_sec REAL DEFAULT 0.0,
                class_id INTEGER NOT NULL,
                class_code TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                box_x1 REAL NOT NULL,
                box_y1 REAL NOT NULL,
                box_x2 REAL NOT NULL,
                box_y2 REAL NOT NULL,
                box_w REAL NOT NULL,
                box_h REAL NOT NULL,
                severity_weight INTEGER DEFAULT 2,
                created_at TEXT NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                weights_path TEXT NOT NULL,
                task TEXT DEFAULT 'detect',
                map50 REAL DEFAULT 0.0,
                map50_95 REAL DEFAULT 0.0,
                registered_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_detections_inspection ON detections(inspection_id);
            CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_code);
            CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
        """)
    return path


def log_inspection(
    data: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> str:
    """Insert or update a road damage inspection record."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    inspection_id = data.get("id") or f"insp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"

    with get_connection(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO inspections (
                id, name, media_type, created_at, duration_sec,
                total_frames, processed_frames, fps, pci_score, pci_rating,
                total_defects, d00_count, d10_count, d20_count, d40_count,
                media_path, output_path, csv_path, model_used, notes
            ) VALUES (
                :id, :name, :media_type, :created_at, :duration_sec,
                :total_frames, :processed_frames, :fps, :pci_score, :pci_rating,
                :total_defects, :d00_count, :d10_count, :d20_count, :d40_count,
                :media_path, :output_path, :csv_path, :model_used, :notes
            )
        """, {
            "id": inspection_id,
            "name": data.get("name", "Unnamed Inspection"),
            "media_type": data.get("media_type", "image"),
            "created_at": data.get("created_at", now_iso),
            "duration_sec": float(data.get("duration_sec", 0.0)),
            "total_frames": int(data.get("total_frames", 1)),
            "processed_frames": int(data.get("processed_frames", 1)),
            "fps": float(data.get("fps", 0.0)),
            "pci_score": float(data.get("pci_score", 100.0)),
            "pci_rating": data.get("pci_rating", "Good"),
            "total_defects": int(data.get("total_defects", 0)),
            "d00_count": int(data.get("d00_count", 0)),
            "d10_count": int(data.get("d10_count", 0)),
            "d20_count": int(data.get("d20_count", 0)),
            "d40_count": int(data.get("d40_count", 0)),
            "media_path": data.get("media_path"),
            "output_path": data.get("output_path"),
            "csv_path": data.get("csv_path"),
            "model_used": data.get("model_used", "YOLO11"),
            "notes": data.get("notes", ""),
        })
    return inspection_id


def log_detections(
    inspection_id: str,
    detections: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> int:
    """Bulk insert detection entries linked to an inspection."""
    if not detections:
        return 0

    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for d in detections:
        code = str(d.get("class_code", d.get("class_name", "D00"))).upper()
        if code not in SEVERITY_WEIGHTS:
            # Map friendly names
            name_lower = code.lower()
            if "pothole" in name_lower or "d40" in name_lower:
                code = "D40"
            elif "alligator" in name_lower or "d20" in name_lower:
                code = "D20"
            elif "transverse" in name_lower or "lateral" in name_lower or "d10" in name_lower:
                code = "D10"
            else:
                code = "D00"

        cls_id = int(d.get("class_id", {"D00": 0, "D10": 1, "D20": 2, "D40": 3}.get(code, 0)))
        nice_name = CLASS_NAMES.get(code, code)
        weight = SEVERITY_WEIGHTS.get(code, 2)

        x1 = float(d.get("box_x1", d.get("xmin", 0.0)))
        y1 = float(d.get("box_y1", d.get("ymin", 0.0)))
        x2 = float(d.get("box_x2", d.get("xmax", 0.0)))
        y2 = float(d.get("box_y2", d.get("ymax", 0.0)))
        bw = float(d.get("box_w", max(0.0, x2 - x1)))
        bh = float(d.get("box_h", max(0.0, y2 - y1)))

        rows.append((
            inspection_id,
            int(d.get("frame_index", d.get("frame", 0))),
            float(d.get("timestamp_sec", d.get("time_sec", 0.0))),
            cls_id,
            code,
            nice_name,
            float(d.get("confidence", 0.0)),
            x1, y1, x2, y2, bw, bh,
            weight,
            now_iso,
        ))

    with get_connection(db_path) as conn:
        conn.executemany("""
            INSERT INTO detections (
                inspection_id, frame_index, timestamp_sec, class_id,
                class_code, class_name, confidence, box_x1, box_y1,
                box_x2, box_y2, box_w, box_h, severity_weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    return len(rows)


def import_csv(
    csv_path: Path,
    inspection_name: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Import an existing detections CSV export into the SQLite database."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    detections = []
    d_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_cls = row.get("class", row.get("class_name", row.get("class_code", "D00")))
            code = raw_cls.upper()
            if code not in d_counts:
                s = raw_cls.lower()
                if "pothole" in s or "d40" in s:
                    code = "D40"
                elif "alligator" in s or "d20" in s:
                    code = "D20"
                elif "transverse" in s or "lateral" in s or "d10" in s:
                    code = "D10"
                else:
                    code = "D00"

            d_counts[code] = d_counts.get(code, 0) + 1
            detections.append({
                "frame_index": int(row.get("frame", row.get("frame_index", 0))),
                "timestamp_sec": float(row.get("time_sec", row.get("timestamp_sec", 0.0))),
                "class_code": code,
                "confidence": float(row.get("confidence", 0.0)),
                "box_x1": float(row.get("x1", row.get("xmin", row.get("box_x1", 0.0)))),
                "box_y1": float(row.get("y1", row.get("ymin", row.get("box_y1", 0.0)))),
                "box_x2": float(row.get("x2", row.get("xmax", row.get("box_x2", 0.0)))),
                "box_y2": float(row.get("y2", row.get("ymax", row.get("box_y2", 0.0)))),
            })

    # Estimate PCI score
    deduct = (d_counts["D00"] * 2) + (d_counts["D10"] * 2) + (d_counts["D20"] * 6) + (d_counts["D40"] * 12)
    pci_score = max(0.0, 100.0 - min(100.0, deduct))
    if pci_score >= 85:
        pci_rating = "Good"
    elif pci_score >= 70:
        pci_rating = "Satisfactory"
    elif pci_score >= 55:
        pci_rating = "Fair"
    elif pci_score >= 40:
        pci_rating = "Poor"
    elif pci_score >= 25:
        pci_rating = "Very Poor"
    else:
        pci_rating = "Serious"

    insp_data = {
        "name": inspection_name or csv_path.stem,
        "media_type": "video" if any(d["frame_index"] > 0 for d in detections) else "image",
        "pci_score": pci_score,
        "pci_rating": pci_rating,
        "total_defects": len(detections),
        "d00_count": d_counts["D00"],
        "d10_count": d_counts["D10"],
        "d20_count": d_counts["D20"],
        "d40_count": d_counts["D40"],
        "csv_path": str(csv_path.resolve()),
    }
    insp_id = log_inspection(insp_data, db_path)
    log_detections(insp_id, detections, db_path)
    return insp_id


def get_inspections(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve list of recent inspections ordered by creation time descending."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM inspections ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [dict(r) for r in cur.fetchall()]


def get_inspection_details(inspection_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve full inspection details including all individual detections."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        insp_cur = conn.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
        insp_row = insp_cur.fetchone()
        if not insp_row:
            return None

        result = dict(insp_row)
        det_cur = conn.execute(
            "SELECT * FROM detections WHERE inspection_id = ? ORDER BY frame_index, timestamp_sec",
            (inspection_id,)
        )
        result["detections"] = [dict(r) for r in det_cur.fetchall()]
        return result


def get_summary_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculate aggregate statistics across all recorded inspections and defects."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        total_inspections = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        total_detections = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        avg_pci = conn.execute("SELECT AVG(pci_score) FROM inspections").fetchone()[0] or 100.0

        # Class breakdown
        class_counts = {
            "D00": 0, "D10": 0, "D20": 0, "D40": 0
        }
        for row in conn.execute("SELECT class_code, COUNT(*) as c FROM detections GROUP BY class_code"):
            code = row["class_code"]
            if code in class_counts:
                class_counts[code] = row["c"]

        return {
            "total_inspections": total_inspections,
            "total_detections": total_detections,
            "average_pci": round(avg_pci, 1),
            "classes": {
                "D00": {"code": "D00", "name": "Longitudinal Crack", "count": class_counts["D00"]},
                "D10": {"code": "D10", "name": "Transverse Crack", "count": class_counts["D10"]},
                "D20": {"code": "D20", "name": "Alligator Crack", "count": class_counts["D20"]},
                "D40": {"code": "D40", "name": "Pothole", "count": class_counts["D40"]},
            },
            "db_path": str((db_path or DEFAULT_DB_PATH).resolve()),
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for database management and inspection querying."""
    parser = argparse.ArgumentParser(
        description="Road Damage Detection SQLite Database CLI Manager."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=os.getenv("ROAD_DAMAGE_DB", str(DEFAULT_DB_PATH)),
        help=f"Path to SQLite database file (default: '{DEFAULT_DB_PATH}').",
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Command: init
    subparsers.add_parser("init", help="Initialize database schema and tables.")

    # Command: stats
    subparsers.add_parser("stats", help="Show summary statistics across all inspections.")

    # Command: list
    list_p = subparsers.add_parser("list", help="List recent inspection records.")
    list_p.add_argument("--limit", type=int, default=20, help="Max records to return.")

    # Command: show
    show_p = subparsers.add_parser("show", help="Show details for a specific inspection.")
    show_p.add_argument("id", type=str, help="Inspection ID to inspect.")

    # Command: import-csv
    import_p = subparsers.add_parser("import-csv", help="Import a detections CSV file.")
    import_p.add_argument("csv_path", type=str, help="Path to CSV file.")
    import_p.add_argument("--name", type=str, default=None, help="Name for this inspection.")

    return parser.parse_args()


def main() -> None:
    """Execute CLI workflow for database initialization and queries."""
    args = parse_args()
    db_path = Path(args.db)

    if args.command == "init" or args.command is None:
        init_db(db_path)
        print(f"[Database] SQLite database initialized at: {db_path.resolve()}")

    elif args.command == "stats":
        stats = get_summary_stats(db_path)
        print("\n" + "=" * 60)
        print(f"ROAD DAMAGE DATABASE SUMMARY: {stats['db_path']}")
        print("=" * 60)
        print(f"Total Inspections : {stats['total_inspections']}")
        print(f"Total Defect Logs : {stats['total_detections']}")
        print(f"Average PCI Score : {stats['average_pci']} / 100.0")
        print("\nDefect Distribution by Class:")
        for code, meta in stats["classes"].items():
            print(f"  [{code}] {meta['name']:<20}: {meta['count']} defects")
        print("=" * 60 + "\n")

    elif args.command == "list":
        records = get_inspections(limit=args.limit, db_path=db_path)
        if not records:
            print("No inspections found in database.")
            return

        print("\n" + "=" * 80)
        print(f"{'ID':<25} {'NAME':<20} {'TYPE':<7} {'DEFECTS':<8} {'PCI':<6} {'RATING'}")
        print("-" * 80)
        for r in records:
            print(f"{r['id']:<25} {r['name'][:18]:<20} {r['media_type']:<7} {r['total_defects']:<8} {r['pci_score']:<6.1f} {r['pci_rating']}")
        print("=" * 80 + "\n")

    elif args.command == "show":
        details = get_inspection_details(args.id, db_path=db_path)
        if not details:
            print(f"Inspection ID '{args.id}' not found.")
            sys.exit(1)

        print(json.dumps(details, indent=2, default=str))

    elif args.command == "import-csv":
        csv_file = Path(args.csv_path)
        insp_id = import_csv(csv_file, inspection_name=args.name, db_path=db_path)
        print(f"[Database] Successfully imported '{csv_file}' -> Inspection ID: {insp_id}")


if __name__ == "__main__":
    main()

