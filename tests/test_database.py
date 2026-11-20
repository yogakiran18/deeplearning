"""Unit tests for SQLite road damage database module."""

import tempfile
from pathlib import Path
import pytest

from src.database import (
    init_db,
    log_inspection,
    log_detections,
    get_inspections,
    get_inspection_details,
    get_summary_stats,
    import_csv,
)


def test_init_db_and_tables():
    """Verify that SQLite tables and indexes are created successfully."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        init_db(db_path)
        assert db_path.exists()
        stats = get_summary_stats(db_path)
        assert stats["total_inspections"] == 0
        assert stats["total_detections"] == 0
    finally:
        db_path.unlink(missing_ok=True)


def test_log_inspection_and_detections():
    """Verify inserting an inspection and querying its detailed detections."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        insp_id = log_inspection({
            "name": "Test Pavement Inspection",
            "media_type": "video",
            "duration_sec": 12.5,
            "total_frames": 375,
            "pci_score": 75.0,
            "pci_rating": "Satisfactory",
            "total_defects": 2,
            "d00_count": 1,
            "d40_count": 1,
        }, db_path=db_path)

        assert insp_id is not None

        num_logged = log_detections(insp_id, [
            {
                "frame_index": 10,
                "timestamp_sec": 0.33,
                "class_code": "D00",
                "confidence": 0.88,
                "box_x1": 100, "box_y1": 150, "box_x2": 200, "box_y2": 250,
            },
            {
                "frame_index": 25,
                "timestamp_sec": 0.83,
                "class_code": "D40",
                "confidence": 0.92,
                "box_x1": 300, "box_y1": 400, "box_x2": 450, "box_y2": 550,
            },
        ], db_path=db_path)

        assert num_logged == 2

        # Verify inspection details
        details = get_inspection_details(insp_id, db_path=db_path)
        assert details is not None
        assert details["name"] == "Test Pavement Inspection"
        assert len(details["detections"]) == 2
        assert details["detections"][0]["class_code"] == "D00"
        assert details["detections"][1]["class_code"] == "D40"

        # Verify summary stats
        stats = get_summary_stats(db_path)
        assert stats["total_inspections"] == 1
        assert stats["total_detections"] == 2
        assert stats["classes"]["D00"]["count"] == 1
        assert stats["classes"]["D40"]["count"] == 1
    finally:
        db_path.unlink(missing_ok=True)

