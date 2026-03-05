"""
init_db.py — ThreatSense AI-DVR
Initializes the SQLite database and creates the incidents table.
Run this ONCE before starting the dashboard: python init_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "threat_logs.db")


def init_database():
    """Create the database and incidents table if they don't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
            camera_name     TEXT    NOT NULL,
            intent_status   TEXT    NOT NULL,
            ollama_reasoning TEXT,
            clip_file_path  TEXT
        )
        """
    )

    conn.commit()
    conn.close()
    print(f"[✔] Database initialised at: {DB_PATH}")
    print("[✔] Table 'incidents' is ready.")


if __name__ == "__main__":
    init_database()
