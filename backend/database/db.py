import sqlite3
from datetime import datetime
import json


class Database:
    def __init__(self, db_path="medical_detector.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                is_medical BOOLEAN NOT NULL,
                medical_confidence REAL NOT NULL,
                is_fake BOOLEAN NOT NULL,
                fake_confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def store_result(self, result: dict):
        """Store analysis result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO analyses (text, is_medical, medical_confidence, is_fake, fake_confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result['text'],
            result['is_medical'],
            result['medical_confidence'],
            result['is_fake'],
            result['fake_confidence'],
            result['timestamp']
        ))

        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        """Get analysis statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total analyses
        cursor.execute('SELECT COUNT(*) FROM analyses')
        total = cursor.fetchone()[0]

        # Medical posts
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE is_medical = 1')
        medical_count = cursor.fetchone()[0]

        # Fake posts
        cursor.execute('SELECT COUNT(*) FROM analyses WHERE is_fake = 1')
        fake_count = cursor.fetchone()[0]

        # Recent analyses (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM analyses 
            WHERE datetime(timestamp) > datetime('now', '-1 day')
        ''')
        recent_count = cursor.fetchone()[0]

        conn.close()

        return {
            'total_analyses': total,
            'medical_posts': medical_count,
            'fake_posts': fake_count,
            'recent_analyses': recent_count,
            'medical_percentage': round((medical_count / max(total, 1)) * 100, 1),
            'fake_percentage': round((fake_count / max(medical_count, 1)) * 100, 1)
        }