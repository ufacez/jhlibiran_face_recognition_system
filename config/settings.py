# config/settings.py
"""
TrackSite Attendance System — Configuration

All settings can be overridden via environment variables or .env file.
"""

import os
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration settings for TrackSite Facial Recognition System"""

    # ── Project / Device ──────────────────────────────────────
    PROJECT_ID: Optional[int] = int(os.getenv('PROJECT_ID', '0')) or None
    DEVICE_NAME: str = os.getenv('DEVICE_NAME', 'TrackSite-Device')

    # ── MySQL Database (Central Server) ───────────────────────
    MYSQL_HOST: str = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT: int = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER: str = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD: str = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE: str = os.getenv('MYSQL_DATABASE', 'construction_management')

    # ── SQLite Database (Local Buffer) ────────────────────────
    SQLITE_PATH: str = os.getenv('SQLITE_PATH', 'data/local.db')

    # ── Face Recognition ──────────────────────────────────────
    FACE_RECOGNITION_TOLERANCE: float = float(os.getenv('FACE_TOLERANCE', '0.5'))
    FACE_DETECTION_MODEL: str = 'hog'  # 'hog' for CPU, 'cnn' for GPU
    MIN_FACE_SIZE: Tuple[int, int] = (50, 50)
    RECOGNITION_SCALE: float = 0.35  # Downscale factor for speed

    # ── Anti-Accidental Safeguards ────────────────────────────
    #   STABILITY_SECONDS   — Worker must stay in detection zone this long
    #   COOLDOWN_SECONDS    — After recording, same worker blocked this long
    #   MIN_WORK_INTERVAL   — Minimum minutes between Time In and Time Out
    STABILITY_SECONDS: float = float(os.getenv('STABILITY_SECONDS', '3.0'))
    COOLDOWN_SECONDS: float = float(os.getenv('COOLDOWN_SECONDS', '60.0'))
    MIN_WORK_INTERVAL_MINUTES: int = int(os.getenv('MIN_WORK_INTERVAL', '30'))
    DUPLICATE_TIMEOUT_SECONDS: int = 30

    # ── Synchronization ──────────────────────────────────────
    SYNC_INTERVAL_SECONDS: int = int(os.getenv('SYNC_INTERVAL', '300'))
    SYNC_API_URL: str = os.getenv('SYNC_API_URL', '')
    SYNC_API_KEY: str = os.getenv('SYNC_API_KEY', '')
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_BACKOFF_MULTIPLIER: int = 2

    # ── Camera / Hardware ─────────────────────────────────────
    CAMERA_INDEX: int = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_RESOLUTION: Tuple[int, int] = (640, 480)
    CAMERA_FRAMERATE: int = 30
    GPIO_TIMEOUT_BUTTON: Optional[int] = None
    GPIO_MODE_LED: Optional[int] = None

    # ── Display ───────────────────────────────────────────────
    FULLSCREEN: bool = os.getenv('FULLSCREEN', 'false').lower() == 'true'
    WINDOW_WIDTH: int = int(os.getenv('WINDOW_WIDTH', '1280'))
    WINDOW_HEIGHT: int = int(os.getenv('WINDOW_HEIGHT', '800'))
    DISPLAY_FEEDBACK_SECONDS: int = 5

    # ── Attendance Logic (legacy compat) ──────────────────────
    AUTO_TIMEOUT_ENABLED: bool = True
    TIMEOUT_CONFIRMATION_SECONDS: int = 5
    DISPLAY_FONT_SCALE: float = 1.5

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = 'logs/system.log'
