import csv
import os
import time
import sqlite3
import threading
from typing import Optional, List, Dict, Any

class EventLogger:
    def __init__(self, csv_path: str, sqlite_path: Optional[str] = None, trap_id: str = 'trap_a'):
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.trap_id = trap_id
        self.lock = threading.Lock()
        
        self.headers = [
            'timestamp', 'trap_id', 'event_id', 'predicted_class', 
            'confidence', 'decision', 'servo_action', 'inference_latency', 
            'actuation_latency', 'fl_round', 'model_version'
        ]
        
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                
        if self.sqlite_path:
            os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS events (
                        timestamp REAL,
                        trap_id TEXT,
                        event_id TEXT,
                        predicted_class TEXT,
                        confidence REAL,
                        decision TEXT,
                        servo_action TEXT,
                        inference_latency REAL,
                        actuation_latency REAL,
                        fl_round INTEGER,
                        model_version TEXT
                    )
                ''')
                conn.commit()

    def log_event(self, event_id: str, predicted_class: str, confidence: float, decision: str, 
                  servo_action: str, inference_latency: float, actuation_latency: float, 
                  fl_round: Optional[int] = None, model_version: Optional[str] = None) -> None:
        timestamp = time.time()
        row = [
            timestamp, self.trap_id, event_id, predicted_class, confidence,
            decision, servo_action, inference_latency, actuation_latency,
            fl_round, model_version
        ]
        
        with self.lock:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            if self.sqlite_path:
                with sqlite3.connect(self.sqlite_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', row)
                    conn.commit()

    def get_recent_events(self, n: int = 10) -> List[Dict[str, Any]]:
        events = []
        with self.lock:
            if self.sqlite_path:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM events ORDER BY timestamp DESC LIMIT ?', (n,))
                    rows = cursor.fetchall()
                    events = [dict(row) for row in rows]
            else:
                if os.path.exists(self.csv_path):
                    with open(self.csv_path, mode='r') as f:
                        reader = list(csv.DictReader(f))
                        events = reader[-n:]
                        events.reverse()
        return events

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            if self.sqlite_path:
                with sqlite3.connect(self.sqlite_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT COUNT(*) FROM events')
                    total = cursor.fetchone()[0]
                    cursor.execute('SELECT AVG(confidence) FROM events')
                    avg_conf = cursor.fetchone()[0]
                    return {'total_events': total, 'avg_confidence': avg_conf or 0.0}
            else:
                if os.path.exists(self.csv_path):
                    with open(self.csv_path, mode='r') as f:
                        reader = list(csv.DictReader(f))
                        total = len(reader)
                        avg_conf = sum(float(r['confidence']) for r in reader) / total if total > 0 else 0.0
                        return {'total_events': total, 'avg_confidence': avg_conf}
                return {'total_events': 0, 'avg_confidence': 0.0}
