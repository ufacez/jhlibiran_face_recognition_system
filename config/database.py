import os
import logging
import mysql.connector
from mysql.connector import Error as MySQLError
import sqlite3
from typing import Optional, List, Dict, Any
from config.settings import Config

logger = logging.getLogger(__name__)


class MySQLDatabase:
    """MySQL database connection manager"""
    
    def __init__(self) -> None:
        self.connection: Optional[mysql.connector.MySQLConnection] = None
        self.is_connected: bool = False
    
    def connect(self) -> bool:
        """Establish MySQL connection"""
        try:
            self.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DATABASE,
                autocommit=True
            )
            self.is_connected = True
            logger.info("MySQL connected")
            return True
        except MySQLError as e:
            logger.error(f"MySQL connection failed: {e}")
            self.is_connected = False
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[int]:
        """Execute INSERT/UPDATE/DELETE"""
        if not self.is_connected or self.connection is None:
            logger.warning("MySQL not connected")
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            last_id = cursor.lastrowid
            cursor.close()
            return last_id
        except MySQLError as e:
            logger.error(f"Query failed: {e}")
            self.is_connected = False
            return None
    
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch multiple rows"""
        if not self.is_connected:
            if not self.connect():
                return []
        
        if self.connection is None:
            return []
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()
            return results if results else []
        except MySQLError as e:
            logger.error(f"Fetch failed: {e}")
            return []
    
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch single row"""
        results = self.fetch_all(query, params)
        return results[0] if results else None
    
    def close(self) -> None:
        """Close connection"""
        if self.connection:
            self.connection.close()
            self.is_connected = False
            logger.info("MySQL closed")


class SQLiteDatabase:
    """Local SQLite for offline buffering"""
    
    def __init__(self) -> None:
        self.db_path: str = Config.SQLITE_PATH
        self._init_database()
    
    def _init_database(self) -> None:
        """Create tables"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Attendance buffer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id INTEGER NOT NULL,
                attendance_date TEXT NOT NULL,
                time_in TEXT,
                time_out TEXT,
                status TEXT DEFAULT 'present',
                hours_worked REAL DEFAULT 0,
                sync_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                synced_at TIMESTAMP
            )
        """)
        
        # Face cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_encodings_cache (
                encoding_id INTEGER PRIMARY KEY,
                worker_id INTEGER NOT NULL,
                encoding_data TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                worker_code TEXT,
                is_active INTEGER DEFAULT 1,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Device configuration cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("SQLite initialized")
    
    def insert_attendance(self, worker_id: int, attendance_date: str, 
                         time_in: Optional[str] = None, 
                         time_out: Optional[str] = None,
                         status: str = 'present') -> int:
        """Insert attendance to buffer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO attendance_buffer 
            (worker_id, attendance_date, time_in, time_out, status)
            VALUES (?, ?, ?, ?, ?)
        """, (worker_id, attendance_date, time_in, time_out, status))
        
        last_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Buffered attendance for worker {worker_id}")
        return last_id
    
    def update_timeout(self, worker_id: int, attendance_date: str, 
                      time_out: str, hours_worked: float) -> bool:
        """Update time-out"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE attendance_buffer 
            SET time_out = ?, hours_worked = ?
            WHERE worker_id = ? AND attendance_date = ? 
            AND time_out IS NULL AND sync_status = 'pending'
        """, (time_out, hours_worked, worker_id, attendance_date))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def get_pending_records(self) -> List[Dict[str, Any]]:
        """Get pending sync records"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM attendance_buffer 
            WHERE sync_status = 'pending'
            ORDER BY created_at ASC
        """)
        
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return records
    
    def mark_synced(self, buffer_id: int) -> None:
        """Mark as synced"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE attendance_buffer 
            SET sync_status = 'synced', synced_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (buffer_id,))
        
        conn.commit()
        conn.close()
    
    def cache_face_encodings(self, encodings: List[Dict[str, Any]]) -> None:
        """Cache face encodings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM face_encodings_cache")
        
        for enc in encodings:
            cursor.execute("""
                INSERT INTO face_encodings_cache 
                (encoding_id, worker_id, encoding_data, first_name, last_name, 
                 worker_code, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                enc['encoding_id'], 
                enc['worker_id'], 
                enc['encoding_data'],
                enc['first_name'], 
                enc['last_name'], 
                enc['worker_code'],
                enc['is_active']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Cached {len(encodings)} encodings")
    
    def get_cached_encodings(self) -> List[Dict[str, Any]]:
        """Get cached encodings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM face_encodings_cache
            WHERE is_active = 1
        """)

        encodings = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return encodings

    def get_today_attendance(
        self, worker_id: int, today: str
    ) -> Optional[Dict[str, Any]]:
        """Get today's attendance from local buffer (offline mode)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id AS attendance_id, worker_id,
                   attendance_date, time_in, time_out,
                   status, hours_worked
            FROM attendance_buffer
            WHERE worker_id = ? AND attendance_date = ?
            ORDER BY created_at DESC LIMIT 1
        """, (worker_id, today))

        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_device_config(self, key: str) -> Optional[str]:
        """Get a device configuration value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM device_config WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def set_device_config(self, key: str, value: str) -> None:
        """Set a device configuration value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO device_config (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        conn.commit()
        conn.close()