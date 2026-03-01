import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """Manages attendance time-in/time-out logic"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.last_scan_cache: Dict[str, datetime] = {}  # Prevent duplicate scans
    
    def log_timein(self, worker_id: int) -> Dict[str, Any]:
        """
        Log worker time-in
        
        Returns:
            {
                'success': bool,
                'action': 'timein' | 'duplicate' | 'error',
                'message': str,
                'worker_info': dict
            }
        """
        today = date.today().isoformat()
        now = datetime.now()
        
        # Check duplicate scan cache
        cache_key = f"{worker_id}_{today}"
        if cache_key in self.last_scan_cache:
            last_scan = self.last_scan_cache[cache_key]
            if (now - last_scan).seconds < Config.DUPLICATE_TIMEOUT_SECONDS:
                return {
                    'success': False,
                    'action': 'duplicate',
                    'message': 'Already scanned recently'
                }
        
        # Check if time-in exists today
        if self.mysql_db.is_connected:
            existing = self.mysql_db.fetch_one("""
                SELECT attendance_id, time_out FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND is_archived = 0
            """, (worker_id, today))
            
            if existing:
                if existing['time_out'] is None:
                    # Already timed-in
                    return {
                        'success': False,
                        'action': 'already_in',
                        'message': 'Already scanned - Attendance recorded',
                        'attendance_id': existing['attendance_id']
                    }
                else:
                    # Already completed
                    return {
                        'success': False,
                        'action': 'completed',
                        'message': 'Already scanned - Time in and out completed'
                    }
        
        # Insert time-in
        time_in = now.strftime('%H:%M:%S')
        time_in_12hr = now.strftime('%I:%M %p')
        attendance_id = None
        
        if self.mysql_db.is_connected:
            # Direct to MySQL with proper transaction
            try:
                query = """
                    INSERT INTO attendance 
                    (worker_id, attendance_date, time_in, status, created_at, updated_at)
                    VALUES (%s, %s, %s, 'present', NOW(), NOW())
                """
                attendance_id = self.mysql_db.execute_query(query, (worker_id, today, time_in))
                
                if attendance_id:
                    # Log to audit_trail for website audit page - Biometric Time In
                    try:
                        import json as _json
                        worker_info = self.mysql_db.fetch_one(
                            "SELECT first_name, last_name, worker_code FROM workers WHERE worker_id = %s",
                            (worker_id,)
                        )
                        w_name = f"{worker_info['first_name']} {worker_info['last_name']}" if worker_info else f'Worker {worker_id}'
                        w_code = worker_info['worker_code'] if worker_info else 'N/A'
                        self.mysql_db.execute_query("""
                            INSERT INTO audit_trail 
                            (user_id, username, user_level, action_type, module, table_name,
                             record_id, record_identifier, old_values, new_values, changes_summary,
                             ip_address, user_agent, severity, is_sensitive, success)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            None, w_name, 'worker', 'time_in', 'attendance', 'attendance',
                            attendance_id, f'{w_name} ({w_code})', None,
                            _json.dumps({'worker_id': worker_id, 'time_in': time_in, 'date': today}),
                            f'Biometric Time In - {w_name} clocked in via facial recognition at {time_in_12hr}',
                            'facial_recognition_system', 'FacialRecognitionDevice', 'low', 0, 1
                        ))
                        logger.info(f"Audit trail logged: Biometric Time In for worker {worker_id}")
                    except Exception as e:
                        logger.error(f"Audit trail failed for biometric time-in: {e}")
            except Exception as e:
                logger.error(f"Time-in insert failed: {e}")
                # Fall through to SQLite buffer
        
        if not attendance_id:
            # Buffer to SQLite if MySQL failed or not connected
            attendance_id = self.sqlite_db.insert_attendance(
                worker_id, today, time_in=time_in
            )
        
        # Update cache
        self.last_scan_cache[cache_key] = now
        
        return {
            'success': True,
            'action': 'timein',
            'message': 'Time-in recorded successfully',
            'attendance_id': attendance_id,
            'time_in': time_in
        }
    
    def log_timeout(self, worker_id: int) -> Dict[str, Any]:
        """
        Log worker time-out
        
        Returns:
            {
                'success': bool,
                'action': 'timeout' | 'no_timein' | 'error',
                'message': str,
                'hours_worked': float
            }
        """
        today = date.today().isoformat()
        now = datetime.now()
        time_out = now.strftime('%H:%M:%S')
        time_out_12hr = now.strftime('%I:%M %p')
        
        # Find today's time-in
        if self.mysql_db.is_connected:
            record = self.mysql_db.fetch_one("""
                SELECT attendance_id, time_in FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND time_out IS NULL AND is_archived = 0
            """, (worker_id, today))
            
            if not record:
                # No time-in found - auto-create one for smooth experience
                logger.warning(f"Worker {worker_id} trying to time-out without time-in. Auto-creating time-in.")
                auto_timein = self.log_timein(worker_id)
                if not auto_timein.get('success'):
                    return {
                        'success': False,
                        'action': 'error',
                        'message': 'Please scan again for time-in first'
                    }
                # Now get the record we just created
                record = self.mysql_db.fetch_one("""
                    SELECT attendance_id, time_in FROM attendance
                    WHERE worker_id = %s AND attendance_date = %s
                    AND time_out IS NULL AND is_archived = 0
                """, (worker_id, today))
                if not record:
                    return {
                        'success': False,
                        'action': 'error',
                        'message': 'System error - please try again'
                    }
            
            # Calculate hours - FIXED: Handle both string and timedelta
            time_in_value = record['time_in']
            
            try:
                if isinstance(time_in_value, str):
                    # It's a string, parse it
                    time_in_dt = datetime.strptime(time_in_value, '%H:%M:%S')
                    time_in_today = datetime.combine(date.today(), time_in_dt.time())
                elif isinstance(time_in_value, timedelta):
                    # It's a timedelta, convert to datetime
                    time_in_today = datetime.combine(date.today(), (datetime.min + time_in_value).time())
                else:
                    # It's already a time object
                    time_in_today = datetime.combine(date.today(), time_in_value)
                
                hours_worked = (now - time_in_today).seconds / 3600
            except Exception as e:
                logger.error(f"Error calculating hours: {e}")
                hours_worked = 0.0
            
            # Update time-out
            self.mysql_db.execute_query("""
                UPDATE attendance 
                SET time_out = %s, hours_worked = %s, updated_at = NOW()
                WHERE attendance_id = %s
            """, (time_out, hours_worked, record['attendance_id']))
            
            # Log to audit_trail for website audit page - Biometric Time Out
            try:
                import json as _json
                worker_info = self.mysql_db.fetch_one(
                    "SELECT first_name, last_name, worker_code FROM workers WHERE worker_id = %s",
                    (worker_id,)
                )
                w_name = f"{worker_info['first_name']} {worker_info['last_name']}" if worker_info else f'Worker {worker_id}'
                w_code = worker_info['worker_code'] if worker_info else 'N/A'
                self.mysql_db.execute_query("""
                    INSERT INTO audit_trail 
                    (user_id, username, user_level, action_type, module, table_name,
                     record_id, record_identifier, old_values, new_values, changes_summary,
                     ip_address, user_agent, severity, is_sensitive, success)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    None, w_name, 'worker', 'time_out', 'attendance', 'attendance',
                    record['attendance_id'], f'{w_name} ({w_code})',
                    _json.dumps({'time_in': str(record['time_in'])}),
                    _json.dumps({'worker_id': worker_id, 'time_out': time_out, 'hours_worked': round(hours_worked, 2), 'date': today}),
                    f'Biometric Time Out - {w_name} clocked out via facial recognition at {time_out_12hr}. Hours: {round(hours_worked, 2)}',
                    'facial_recognition_system', 'FacialRecognitionDevice', 'low', 0, 1
                ))
                logger.info(f"Audit trail logged: Biometric Time Out for worker {worker_id}")
            except Exception as e:
                logger.error(f"Audit trail failed for biometric time-out: {e}")
        else:
            # Buffer to SQLite
            hours_worked = 8.0  # Default estimate
            success = self.sqlite_db.update_timeout(
                worker_id, today, time_out, hours_worked
            )
            
            if not success:
                # Auto-create time-in for offline mode
                logger.warning(f"Worker {worker_id} trying to time-out without time-in (offline). Auto-creating.")
                return {
                    'success': False,
                    'action': 'error',
                    'message': 'Please scan again for time-in first'
                }
        
        return {
            'success': True,
            'action': 'timeout',
            'message': 'Time-out recorded successfully',
            'time_out': time_out,
            'hours_worked': round(hours_worked, 2)
        }