"""
Attendance Logger — Auto Time-In / Time-Out with Anti-Accidental Safeguards

Fully automatic attendance processing:
  • No record today        → TIME IN
  • Has time_in, no out    → TIME OUT (if min interval met)
  • Has time_in, too soon  → blocked (min interval not met)
  • Both in and out        → ALREADY COMPLETED
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any

from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """Manages attendance with automatic time-in/time-out detection."""

    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   PUBLIC — Auto-detect and process
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def process_attendance(self, worker_id: int) -> Dict[str, Any]:
        """
        Auto-detect time-in or time-out and process.

        Returns dict with:
            success  — bool
            action   — 'timein' | 'timeout' | 'too_soon' | 'completed' | 'error'
            message  — human-readable status
        """
        today_str = date.today().isoformat()
        now = datetime.now()

        try:
            existing = self._get_today_record(worker_id, today_str)
        except Exception as e:
            logger.error(f"Error getting today's record: {e}")
            return {
                'success': False,
                'action': 'error',
                'message': 'Database error — please try again',
            }

        # ── No record → TIME IN ───────────────────────
        if existing is None:
            return self._record_timein(worker_id, today_str, now)

        # ── Has time_in, no time_out ──────────────────
        if existing.get('time_out') is None:
            time_in_dt = self._parse_time_value(existing['time_in'])
            time_in_today = datetime.combine(date.today(), time_in_dt.time())
            elapsed_seconds = (now - time_in_today).total_seconds()
            elapsed_minutes = elapsed_seconds / 60

            if elapsed_minutes < Config.MIN_WORK_INTERVAL_MINUTES:
                required = Config.MIN_WORK_INTERVAL_MINUTES
                return {
                    'success': False,
                    'action': 'too_soon',
                    'message': (
                        f'Min {required} min before Time Out '
                        f'({int(elapsed_minutes)} min elapsed)'
                    ),
                }

            return self._record_timeout(
                worker_id, today_str, now, existing, time_in_today)

        # ── Both time_in and time_out present ─────────
        return {
            'success': False,
            'action': 'completed',
            'message': 'Attendance completed for today',
        }

    # Legacy compatible methods used by main_opencv.py
    def log_timein(self, worker_id: int) -> Dict[str, Any]:
        """Legacy: force time-in regardless of existing record."""
        today_str = date.today().isoformat()
        now = datetime.now()

        existing = self._get_today_record(worker_id, today_str)
        if existing is not None:
            if existing.get('time_out') is None:
                return {
                    'success': False,
                    'action': 'already_in',
                    'message': 'Already scanned — attendance recorded',
                    'attendance_id': existing.get('attendance_id'),
                }
            else:
                return {
                    'success': False,
                    'action': 'completed',
                    'message': 'Time in and out completed',
                }

        return self._record_timein(worker_id, today_str, now)

    def log_timeout(self, worker_id: int) -> Dict[str, Any]:
        """Legacy: force time-out."""
        today_str = date.today().isoformat()
        now = datetime.now()

        existing = self._get_today_record(worker_id, today_str)
        if existing is None:
            return {
                'success': False,
                'action': 'error',
                'message': 'No time-in found for today',
            }

        if existing.get('time_out') is not None:
            return {
                'success': False,
                'action': 'completed',
                'message': 'Already completed',
            }

        time_in_dt = self._parse_time_value(existing['time_in'])
        time_in_today = datetime.combine(date.today(), time_in_dt.time())
        return self._record_timeout(
            worker_id, today_str, now, existing, time_in_today)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   INTERNAL — Database operations
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _get_today_record(
        self, worker_id: int, today: str
    ) -> Optional[Dict[str, Any]]:
        """Get today's attendance record (MySQL first, fallback SQLite)."""
        if self.mysql_db and self.mysql_db.is_connected:
            return self.mysql_db.fetch_one("""
                SELECT attendance_id, time_in, time_out
                FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND is_archived = 0
            """, (worker_id, today))
        else:
            return self.sqlite_db.get_today_attendance(worker_id, today)

    def _record_timein(
        self, worker_id: int, today: str, now: datetime
    ) -> Dict[str, Any]:
        """Insert a time-in record."""
        time_in = now.strftime('%H:%M:%S')
        time_in_12hr = now.strftime('%I:%M %p')
        attendance_id = None

        if self.mysql_db and self.mysql_db.is_connected:
            try:
                attendance_id = self.mysql_db.execute_query("""
                    INSERT INTO attendance
                    (worker_id, attendance_date, time_in, status,
                     created_at, updated_at)
                    VALUES (%s, %s, %s, 'present', NOW(), NOW())
                """, (worker_id, today, time_in))

                if attendance_id:
                    self._log_audit(
                        worker_id, attendance_id,
                        'time_in', time_in_12hr, today)
            except Exception as e:
                logger.error(f"MySQL time-in failed: {e}")
                attendance_id = None

        if not attendance_id:
            attendance_id = self.sqlite_db.insert_attendance(
                worker_id, today, time_in=time_in)

        logger.info(f"TIME IN: Worker {worker_id} at {time_in_12hr}")

        return {
            'success': True,
            'action': 'timein',
            'message': 'Time-in recorded',
            'time_in': time_in_12hr,
            'attendance_id': attendance_id,
        }

    def _record_timeout(
        self, worker_id: int, today: str, now: datetime,
        existing: Dict[str, Any], time_in_today: datetime
    ) -> Dict[str, Any]:
        """Update an existing record with time-out."""
        time_out = now.strftime('%H:%M:%S')
        time_out_12hr = now.strftime('%I:%M %p')
        hours_worked = round(
            (now - time_in_today).total_seconds() / 3600, 2)

        if self.mysql_db and self.mysql_db.is_connected:
            try:
                self.mysql_db.execute_query("""
                    UPDATE attendance
                    SET time_out = %s, hours_worked = %s, updated_at = NOW()
                    WHERE attendance_id = %s
                """, (time_out, hours_worked, existing['attendance_id']))

                self._log_audit(
                    worker_id, existing['attendance_id'],
                    'time_out', time_out_12hr, today, hours_worked)
            except Exception as e:
                logger.error(f"MySQL time-out failed: {e}")
                self.sqlite_db.update_timeout(
                    worker_id, today, time_out, hours_worked)
        else:
            success = self.sqlite_db.update_timeout(
                worker_id, today, time_out, hours_worked)
            if not success:
                logger.error(
                    f"SQLite time-out update failed for worker {worker_id}")

        logger.info(
            f"TIME OUT: Worker {worker_id} at {time_out_12hr} "
            f"(Hours: {hours_worked})")

        return {
            'success': True,
            'action': 'timeout',
            'message': 'Time-out recorded',
            'time_out': time_out_12hr,
            'hours_worked': hours_worked,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _parse_time_value(self, value) -> datetime:
        """Parse time value from MySQL (string | timedelta | time)."""
        try:
            if isinstance(value, str):
                return datetime.strptime(value, '%H:%M:%S')
            elif isinstance(value, timedelta):
                return datetime.min + value
            else:
                return datetime.combine(date.today(), value)
        except Exception:
            return datetime.now()

    def _log_audit(
        self, worker_id: int, attendance_id: int,
        action_type: str, time_str: str, today: str,
        hours: float = None
    ):
        """Log to audit_trail table for the website audit page."""
        try:
            worker_info = self.mysql_db.fetch_one(
                "SELECT first_name, last_name, worker_code "
                "FROM workers WHERE worker_id = %s",
                (worker_id,))

            if not worker_info:
                return

            w_name = (f"{worker_info['first_name']} "
                      f"{worker_info['last_name']}")
            w_code = worker_info['worker_code']

            if action_type == 'time_in':
                summary = (
                    f'Biometric Time In — {w_name} clocked in via '
                    f'facial recognition at {time_str}')
                new_values = json.dumps({
                    'worker_id': worker_id,
                    'time_in': time_str,
                    'date': today,
                })
            else:
                summary = (
                    f'Biometric Time Out — {w_name} clocked out via '
                    f'facial recognition at {time_str}. '
                    f'Hours: {hours}')
                new_values = json.dumps({
                    'worker_id': worker_id,
                    'time_out': time_str,
                    'hours_worked': hours,
                    'date': today,
                })

            self.mysql_db.execute_query("""
                INSERT INTO audit_trail
                (user_id, username, user_level, action_type, module,
                 table_name, record_id, record_identifier, new_values,
                 changes_summary, ip_address, user_agent, severity,
                 is_sensitive, success)
                VALUES (%s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s)
            """, (
                None, w_name, 'worker', action_type, 'attendance',
                'attendance', attendance_id,
                f'{w_name} ({w_code})', new_values,
                summary, 'facial_recognition_system',
                'FacialRecognitionDevice', 'low', 0, 1,
            ))
            logger.info(
                f"Audit logged: {action_type} for worker {worker_id}")
        except Exception as e:
            logger.error(f"Audit trail error: {e}")
