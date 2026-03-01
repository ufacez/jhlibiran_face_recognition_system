import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)

# Optional HTTP sync
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class SyncManager:
    """Synchronizes offline buffer with central database.

    Supports two sync modes:
      1. Direct MySQL — when the device is on the same LAN / VPN
      2. HTTP API     — when only internet (4G / hotspot) is available
    """

    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.retry_count: Dict[str, int] = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   sync_all — entry point
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def sync_all(self) -> Dict[str, int]:
        """Sync all pending records. Returns {synced, failed, pending}."""

        # Prefer direct MySQL
        if self.mysql_db and self.mysql_db.is_connected:
            return self._sync_via_mysql()

        # Try reconnecting MySQL
        if self.mysql_db:
            try:
                if self.mysql_db.connect():
                    return self._sync_via_mysql()
            except Exception:
                pass

        # Fallback to HTTP API
        if Config.SYNC_API_URL and HAS_REQUESTS:
            return self._sync_via_http()

        logger.warning("No sync path available (MySQL offline, no API URL)")
        pending = self.sqlite_db.get_pending_records()
        return {'synced': 0, 'failed': 0, 'pending': len(pending)}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   MySQL sync
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _sync_via_mysql(self) -> Dict[str, int]:
        pending = self.sqlite_db.get_pending_records()
        synced = 0
        failed = 0

        for record in pending:
            bid = record['id']
            rkey = f"buffer_{bid}"

            if self.retry_count.get(rkey, 0) >= Config.MAX_RETRY_ATTEMPTS:
                failed += 1
                continue

            if self._sync_record_mysql(record):
                self.sqlite_db.mark_synced(bid)
                synced += 1
                self.retry_count.pop(rkey, None)
            else:
                failed += 1
                self.retry_count[rkey] = self.retry_count.get(rkey, 0) + 1

        remaining = len(pending) - synced - failed
        if synced:
            logger.info(
                f"MySQL sync: {synced} synced, {failed} failed, "
                f"{remaining} pending")
        return {'synced': synced, 'failed': failed, 'pending': remaining}

    def _sync_record_mysql(self, record: Dict[str, Any]) -> bool:
        try:
            existing = self.mysql_db.fetch_one("""
                SELECT attendance_id FROM attendance
                WHERE worker_id = %s AND attendance_date = %s
                AND is_archived = 0
            """, (record['worker_id'], record['attendance_date']))

            if existing:
                if record['time_out']:
                    self.mysql_db.execute_query("""
                        UPDATE attendance
                        SET time_out = %s, hours_worked = %s,
                            updated_at = NOW()
                        WHERE attendance_id = %s AND time_out IS NULL
                    """, (record['time_out'], record['hours_worked'],
                          existing['attendance_id']))
            else:
                self.mysql_db.execute_query("""
                    INSERT INTO attendance
                    (worker_id, attendance_date, time_in, time_out,
                     status, hours_worked)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    record['worker_id'],
                    record['attendance_date'],
                    record['time_in'],
                    record['time_out'],
                    record['status'],
                    record['hours_worked'],
                ))
            return True
        except Exception as e:
            logger.error(f"MySQL sync failed for buffer {record['id']}: {e}")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   HTTP API sync (4G / hotspot / remote)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _sync_via_http(self) -> Dict[str, int]:
        pending = self.sqlite_db.get_pending_records()
        if not pending:
            return {'synced': 0, 'failed': 0, 'pending': 0}

        synced = 0
        failed = 0

        # Build batch payload
        records_payload = []
        for record in pending:
            records_payload.append({
                'buffer_id': record['id'],
                'worker_id': record['worker_id'],
                'attendance_date': record['attendance_date'],
                'time_in': record['time_in'],
                'time_out': record['time_out'],
                'status': record['status'],
                'hours_worked': record['hours_worked'],
            })

        try:
            resp = requests.post(
                Config.SYNC_API_URL,
                json={
                    'action': 'sync_attendance',
                    'api_key': Config.SYNC_API_KEY,
                    'device_name': Config.DEVICE_NAME,
                    'project_id': Config.PROJECT_ID,
                    'records': records_payload,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get('success'):
                    synced_ids = data.get('synced_ids', [])
                    for bid in synced_ids:
                        self.sqlite_db.mark_synced(bid)
                        synced += 1
                    failed = len(pending) - synced
                    logger.info(f"HTTP sync: {synced} synced")
                else:
                    failed = len(pending)
                    logger.warning(
                        f"HTTP sync rejected: {data.get('message')}")
            else:
                failed = len(pending)
                logger.error(
                    f"HTTP sync error: status {resp.status_code}")
        except Exception as e:
            failed = len(pending)
            logger.error(f"HTTP sync exception: {e}")

        remaining = len(pending) - synced - failed
        return {'synced': synced, 'failed': failed, 'pending': remaining}