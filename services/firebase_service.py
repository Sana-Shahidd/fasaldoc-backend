"""
Scan history service — SQLite backend (free, no external service needed).
Replaces Firebase Firestore. Data stored in history.db inside the container.
"""

import sqlite3
import structlog
from datetime import datetime, timezone
from pathlib import Path

logger = structlog.get_logger()

DB_PATH = Path(__file__).parent.parent / "history.db"


def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            scan_id   TEXT PRIMARY KEY,
            user_id   TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            label_key TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            lat  REAL,
            lng  REAL,
            image_thumbnail_url TEXT
        )
    """)
    conn.commit()
    return conn


async def save_scan(
    user_id: str,
    label_key: str,
    disease_name: str,
    confidence: float,
    lat: float = None,
    lng: float = None,
    image_thumbnail_url: str = None,
) -> str | None:
    try:
        import uuid
        scan_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        with _get_conn() as conn:
            conn.execute(
                """INSERT INTO scans
                   (scan_id, user_id, timestamp, label_key, disease_name,
                    confidence, lat, lng, image_thumbnail_url)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (scan_id, user_id, ts, label_key, disease_name,
                 confidence, lat, lng, image_thumbnail_url),
            )
        logger.info("Scan saved", scan_id=scan_id, user_id=user_id)
        return scan_id
    except Exception as e:
        logger.error("Failed to save scan", error=str(e))
        return None


async def get_history(user_id: str, limit: int = 20) -> list:
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM scans WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error("Failed to fetch history", error=str(e))
        return []
