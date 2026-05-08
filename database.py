import sqlite3
import datetime
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            total_vehicles INTEGER DEFAULT 0,
            chart_data TEXT DEFAULT '{}'
        )
    ''')
    # Add new columns if they don't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE history ADD COLUMN total_vehicles INTEGER DEFAULT 0')
    except:
        pass
    try:
        cursor.execute('ALTER TABLE history ADD COLUMN chart_data TEXT DEFAULT "{}"')
    except:
        pass
    conn.commit()
    conn.close()

def add_history(filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('INSERT INTO history (filename, timestamp) VALUES (?, ?)', (filename, timestamp))
    history_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return history_id

def update_history(history_id, total_vehicles, chart_data_dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE history SET total_vehicles = ?, chart_data = ? WHERE id = ?',
        (total_vehicles, json.dumps(chart_data_dict), history_id)
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, timestamp, total_vehicles FROM history ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "filename": row[1], "timestamp": row[2], "total_vehicles": row[3] or 0} for row in rows]

def get_history_by_id(history_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, timestamp, total_vehicles, chart_data FROM history WHERE id = ?', (history_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "filename": row[1],
            "timestamp": row[2],
            "total_vehicles": row[3] or 0,
            "chart_data": json.loads(row[4]) if row[4] else {}
        }
    return None
