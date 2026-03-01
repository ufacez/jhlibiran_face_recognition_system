"""
TrackSite Attendance System — Tkinter GUI Application

Standalone facial recognition attendance for construction sites.

Features:
  • Live camera feed with face detection & landmarks
  • Automatic Time In / Time Out (no manual toggle)
  • Anti-accidental safeguards (stability check, cooldown, min interval)
  • Offline-first local SQLite with automatic sync
  • Per-project worker filtering

Controls:
  F   — Toggle fullscreen
  R   — Reload face data
  Q / Esc — Exit

Usage:
  python main.py
  pythonw main.py        (no console window)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import threading
import time
import sys
import os
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List

# Ensure correct working directory (for .env loading)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer
from models.attendance_logger import AttendanceLogger
from models.sync_manager import SyncManager
from utils.camera import Camera

# ── Logging ──────────────────────────────────────────────────
os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
os.makedirs('data', exist_ok=True)

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Design Tokens — Dark theme inspired by TrackSite branding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG         = '#0D1117'
CARD_BG    = '#161B22'
HEADER_BG  = '#010409'
GOLD       = '#DAA520'
DARK_GOLD  = '#B8860B'
SUCCESS    = '#3FB950'
DANGER     = '#F85149'
WARNING    = '#D29922'
PRIMARY    = '#58A6FF'
TEXT       = '#E6EDF3'
TEXT_SEC   = '#8B949E'
SEPARATOR  = '#30363D'
FONT       = 'Segoe UI'
CAMERA_BG  = '#000000'

STATUS_COLORS = {
    'timein':    {'bg': '#0D2818', 'fg': SUCCESS,  'icon': '✓'},
    'timeout':   {'bg': '#0D1B2A', 'fg': PRIMARY,  'icon': '✓'},
    'cooldown':  {'bg': '#1A1A1A', 'fg': TEXT_SEC, 'icon': '⏳'},
    'too_soon':  {'bg': '#2A1F00', 'fg': WARNING,  'icon': '⚠'},
    'completed': {'bg': '#1A1A1A', 'fg': TEXT_SEC, 'icon': '✓'},
    'error':     {'bg': '#2A0D0D', 'fg': DANGER,   'icon': '✕'},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stability Tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StabilityTracker:
    """Tracks how long a recognized face stays in the detection zone."""

    def __init__(self):
        self.worker_id: Optional[int] = None
        self.worker_name: str = ''
        self.worker_code: str = ''
        self.first_seen: Optional[float] = None
        self.last_seen: Optional[float] = None
        self.frame_count: int = 0

    def update(self, worker_id: int, name: str = '', code: str = ''):
        now = time.time()
        if worker_id != self.worker_id:
            self.worker_id = worker_id
            self.worker_name = name
            self.worker_code = code
            self.first_seen = now
            self.frame_count = 0
        self.last_seen = now
        self.frame_count += 1

    def reset(self):
        self.worker_id = None
        self.worker_name = ''
        self.worker_code = ''
        self.first_seen = None
        self.last_seen = None
        self.frame_count = 0

    @property
    def duration(self) -> float:
        if self.first_seen and self.last_seen:
            return self.last_seen - self.first_seen
        return 0.0

    def is_stable(self) -> bool:
        return self.duration >= Config.STABILITY_SECONDS

    @property
    def progress(self) -> float:
        if Config.STABILITY_SECONDS <= 0:
            return 1.0
        return min(1.0, self.duration / Config.STABILITY_SECONDS)

    @property
    def is_active(self) -> bool:
        return self.worker_id is not None and self.last_seen is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AttendanceApp:
    """Tkinter GUI for the TrackSite biometric attendance system."""

    def __init__(self):
        logger.info("Starting TrackSite Attendance System (Tkinter GUI)...")

        self.root = tk.Tk()
        self.root.title("TrackSite — Attendance System")
        self.root.configure(bg=BG)
        self.root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")
        self.root.minsize(960, 600)

        # ── State ─────────────────────────────────────────
        self.mysql_db: Optional[MySQLDatabase] = None
        self.sqlite_db: Optional[SQLiteDatabase] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.attendance_logger: Optional[AttendanceLogger] = None
        self.sync_manager: Optional[SyncManager] = None
        self.camera: Optional[Camera] = None

        self.is_running = False
        self.is_fullscreen = False

        # Recognition (shared with background thread)
        self.current_faces: List[Dict[str, Any]] = []
        self.faces_lock = threading.Lock()

        # Stability & cooldown
        self.stability = StabilityTracker()
        self.cooldowns: Dict[int, float] = {}
        self.attendance_triggered_for: Optional[int] = None

        # Notification
        self.notification: Optional[Dict[str, Any]] = None
        self.notification_expiry: float = 0

        # UI refs
        self.photo_image = None
        self.project_name = ''
        self.encoding_count = 0

        # ── Build UI ──────────────────────────────────────
        self._build_header()
        self._build_main_area()
        self._build_footer()

        # ── Initialize system ─────────────────────────────
        self._initialize()

        # ── Center window ─────────────────────────────────
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        if Config.FULLSCREEN:
            self._toggle_fullscreen()

        # ── Key bindings ──────────────────────────────────
        self.root.bind('<Escape>', lambda e: self._handle_escape())
        self.root.bind('<q>', lambda e: self._shutdown())
        self.root.bind('<f>', lambda e: self._toggle_fullscreen())
        self.root.bind('<r>', lambda e: self._reload_encodings())
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown)

        # ── Start ─────────────────────────────────────────
        self.is_running = True

        self.recognition_thread = threading.Thread(
            target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()

        self.sync_thread = threading.Thread(
            target=self._sync_worker, daemon=True)
        self.sync_thread.start()

        self._camera_loop()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   HEADER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_header(self):
        header = tk.Frame(self.root, bg=HEADER_BG, height=52)
        header.pack(fill='x')
        header.pack_propagate(False)

        # Left — branding
        left = tk.Frame(header, bg=HEADER_BG)
        left.pack(side='left', padx=16)

        tk.Frame(left, bg=GOLD, width=4, height=24).pack(
            side='left', padx=(0, 10))
        tk.Label(left, text="TrackSite", font=(FONT, 14, 'bold'),
                 fg=GOLD, bg=HEADER_BG).pack(side='left')

        self.project_label = tk.Label(
            left, text="  Attendance System",
            font=(FONT, 12), fg=TEXT_SEC, bg=HEADER_BG)
        self.project_label.pack(side='left', padx=(8, 0))

        # Right — time + status
        right = tk.Frame(header, bg=HEADER_BG)
        right.pack(side='right', padx=16)

        self.time_label = tk.Label(
            right, text="", font=(FONT, 12, 'bold'),
            fg=TEXT, bg=HEADER_BG)
        self.time_label.pack(side='right', padx=(12, 0))

        self.connection_label = tk.Label(
            right, text="● Offline", font=(FONT, 10),
            fg=DANGER, bg=HEADER_BG)
        self.connection_label.pack(side='right')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   MAIN AREA (Camera + Info Panel)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_main_area(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=12, pady=(8, 4))

        # Left: Camera feed (fills available space)
        camera_frame = tk.Frame(
            main, bg=CAMERA_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        camera_frame.pack(side='left', fill='both', expand=True)

        self.camera_label = tk.Label(camera_frame, bg=CAMERA_BG)
        self.camera_label.pack(fill='both', expand=True)

        self.camera_placeholder = tk.Label(
            camera_frame,
            text="📷\nInitializing Camera...",
            font=(FONT, 16), fg=TEXT_SEC, bg=CAMERA_BG,
            justify='center')
        self.camera_placeholder.place(relx=0.5, rely=0.5, anchor='center')

        # Right: Info panel
        info_panel = tk.Frame(main, bg=BG, width=320)
        info_panel.pack(side='right', fill='y', padx=(8, 0))
        info_panel.pack_propagate(False)

        self._build_info_panel(info_panel)

    def _build_info_panel(self, parent):
        """Build the right-side panel with stability, notifications, and stats."""

        # ── Detection Status Card ─────────────────────────
        stab_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        stab_card.pack(fill='x', pady=(0, 8))

        tk.Label(
            stab_card, text="DETECTION STATUS",
            font=(FONT, 9, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w').pack(fill='x', padx=14, pady=(12, 4))

        self.stab_name_label = tk.Label(
            stab_card, text="No face detected",
            font=(FONT, 13, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w')
        self.stab_name_label.pack(fill='x', padx=14)

        # Progress bar
        bar_container = tk.Frame(stab_card, bg=SEPARATOR, height=8)
        bar_container.pack(fill='x', padx=14, pady=(8, 4))
        bar_container.pack_propagate(False)

        self.stab_bar_fill = tk.Frame(
            bar_container, bg=TEXT_SEC, height=8)
        self.stab_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self.stab_text_label = tk.Label(
            stab_card, text="Waiting for worker...",
            font=(FONT, 10), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w')
        self.stab_text_label.pack(fill='x', padx=14, pady=(0, 12))

        # ── Last Action Card ──────────────────────────────
        self.notif_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        self.notif_card.pack(fill='x', pady=(0, 8))

        self.notif_header_label = tk.Label(
            self.notif_card, text="LAST ACTION",
            font=(FONT, 9, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w')
        self.notif_header_label.pack(fill='x', padx=14, pady=(12, 4))

        self.notif_icon_label = tk.Label(
            self.notif_card, text="—",
            font=(FONT, 28), fg=TEXT_SEC, bg=CARD_BG)
        self.notif_icon_label.pack(padx=14, pady=(4, 0))

        self.notif_title_label = tk.Label(
            self.notif_card, text="No actions yet",
            font=(FONT, 13, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w', wraplength=280)
        self.notif_title_label.pack(fill='x', padx=14)

        self.notif_detail_label = tk.Label(
            self.notif_card, text="System ready",
            font=(FONT, 10), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w', wraplength=280)
        self.notif_detail_label.pack(fill='x', padx=14, pady=(2, 12))

        # ── Today's Summary Card ──────────────────────────
        summary_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        summary_card.pack(fill='x', pady=(0, 8))

        tk.Label(
            summary_card, text="TODAY'S SUMMARY",
            font=(FONT, 9, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w').pack(fill='x', padx=14, pady=(12, 8))

        stats_frame = tk.Frame(summary_card, bg=CARD_BG)
        stats_frame.pack(fill='x', padx=14, pady=(0, 12))

        self.present_label = tk.Label(
            stats_frame, text="0", font=(FONT, 24, 'bold'),
            fg=SUCCESS, bg=CARD_BG)
        self.present_label.pack(side='left')
        tk.Label(
            stats_frame, text=" present",
            font=(FONT, 12), fg=TEXT_SEC, bg=CARD_BG
        ).pack(side='left', padx=(4, 20))

        self.completed_label = tk.Label(
            stats_frame, text="0", font=(FONT, 24, 'bold'),
            fg=PRIMARY, bg=CARD_BG)
        self.completed_label.pack(side='left')
        tk.Label(
            stats_frame, text=" completed",
            font=(FONT, 12), fg=TEXT_SEC, bg=CARD_BG
        ).pack(side='left')

        # ── Controls Help Card ────────────────────────────
        help_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        help_card.pack(fill='x', side='bottom')

        tk.Label(
            help_card, text="CONTROLS",
            font=(FONT, 9, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w').pack(fill='x', padx=14, pady=(12, 6))

        for key, desc in [("F", "Toggle fullscreen"),
                          ("R", "Reload face data"),
                          ("Q / Esc", "Exit application")]:
            row = tk.Frame(help_card, bg=CARD_BG)
            row.pack(fill='x', padx=14, pady=1)
            tk.Label(
                row, text=f"  {key}  ",
                font=(FONT, 9, 'bold'), fg=CARD_BG, bg=TEXT_SEC
            ).pack(side='left')
            tk.Label(
                row, text=f"  {desc}",
                font=(FONT, 9), fg=TEXT_SEC, bg=CARD_BG
            ).pack(side='left')

        tk.Frame(help_card, bg=CARD_BG, height=12).pack()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   FOOTER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_footer(self):
        footer = tk.Frame(self.root, bg=HEADER_BG, height=32)
        footer.pack(fill='x')
        footer.pack_propagate(False)

        self.date_label = tk.Label(
            footer, text="", font=(FONT, 10),
            fg=TEXT_SEC, bg=HEADER_BG)
        self.date_label.pack(side='left', padx=16)

        self.sync_label = tk.Label(
            footer, text="Sync: —", font=(FONT, 10),
            fg=TEXT_SEC, bg=HEADER_BG)
        self.sync_label.pack(side='left', padx=(20, 0))

        self.encoding_label = tk.Label(
            footer, text="Faces: 0", font=(FONT, 10),
            fg=TEXT_SEC, bg=HEADER_BG)
        self.encoding_label.pack(side='right', padx=16)

        self.device_label = tk.Label(
            footer, text=Config.DEVICE_NAME, font=(FONT, 10),
            fg=TEXT_SEC, bg=HEADER_BG)
        self.device_label.pack(side='right', padx=(0, 20))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _initialize(self):
        """Initialize all system components."""
        logger.info("Initializing components...")

        # Databases
        self.mysql_db = MySQLDatabase()
        self.sqlite_db = SQLiteDatabase()

        mysql_ok = False
        try:
            mysql_ok = self.mysql_db.connect()
        except Exception:
            pass

        if mysql_ok:
            self.connection_label.config(text="● Online", fg=SUCCESS)
            logger.info("MySQL connected")
        else:
            self.connection_label.config(text="● Offline", fg=WARNING)
            logger.warning("MySQL unavailable — running offline")

        # Load project info
        self._load_project_info()

        # Core components
        self.face_recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
        self.attendance_logger = AttendanceLogger(
            self.mysql_db, self.sqlite_db)
        self.sync_manager = SyncManager(self.mysql_db, self.sqlite_db)

        # Load face encodings
        try:
            self.encoding_count = self.face_recognizer.load_encodings(
                project_id=Config.PROJECT_ID)
        except Exception as e:
            logger.error(f"Failed to load encodings: {e}")
            self.encoding_count = 0

        self.encoding_label.config(text=f"Faces: {self.encoding_count}")
        logger.info(f"Loaded {self.encoding_count} face encodings")

        # Camera
        self.camera = Camera(Config.CAMERA_INDEX)
        if not self.camera.initialize():
            logger.error("Camera initialization failed!")
            self.camera_placeholder.config(
                text="❌\nCamera Error\n\nCheck connection and restart",
                fg=DANGER)
            return

        self.camera.set_resolution(*Config.CAMERA_RESOLUTION)
        self.camera.set_fps(Config.CAMERA_FRAMERATE)

        # Hide placeholder
        self.camera_placeholder.place_forget()

        # Initial summary
        self._update_summary()

        logger.info("Initialization complete")

    def _load_project_info(self):
        """Load project name from database."""
        if (Config.PROJECT_ID and self.mysql_db
                and self.mysql_db.is_connected):
            project = self.mysql_db.fetch_one(
                "SELECT project_name FROM projects WHERE project_id = %s",
                (Config.PROJECT_ID,))
            if project:
                self.project_name = project['project_name']
                self.project_label.config(
                    text=f"  │  {self.project_name}")
                return

        self.project_label.config(text="  Attendance System")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   CAMERA LOOP (Main Thread — ~30 fps)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _camera_loop(self):
        if not self.is_running:
            return

        if self.camera:
            ret, frame = self.camera.read_frame()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                display = frame.copy()

                # Read latest recognition results
                with self.faces_lock:
                    faces = list(self.current_faces)

                # Draw face overlays
                self._draw_faces(display, faces)

                # Update stability tracking
                self._update_stability(faces)

                # Convert BGR → RGB → PIL → PhotoImage
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                try:
                    cw = self.camera_label.winfo_width()
                    ch = self.camera_label.winfo_height()
                    if cw > 10 and ch > 10:
                        fh, fw = frame.shape[:2]
                        scale = min(cw / fw, ch / fh)
                        nw, nh = int(fw * scale), int(fh * scale)
                        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)
                except Exception:
                    pass

                self.photo_image = ImageTk.PhotoImage(pil_img)
                try:
                    self.camera_label.config(image=self.photo_image)
                except tk.TclError:
                    return

        # Update clock
        self._update_clock()

        self.root.after(33, self._camera_loop)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   FACE DRAWING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _draw_faces(self, frame, faces):
        """Draw face boxes, names, and landmark mesh."""
        for face in faces:
            top, right, bottom, left = face['box']
            worker_id = face.get('worker_id')
            name = face.get('name', 'Unknown')
            landmarks = face.get('landmarks', {})

            if worker_id:
                in_cooldown = (
                    worker_id in self.cooldowns
                    and time.time() < self.cooldowns[worker_id])
                color = (150, 150, 150) if in_cooldown else (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)

            # Name label
            label = name if worker_id else "Unknown"
            label_y = max(30, top - 12)
            # Shadow
            cv2.putText(frame, label, (left + 2, label_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, label, (left, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Facial landmarks mesh
            for feature_name, points in landmarks.items():
                for i in range(len(points) - 1):
                    cv2.line(
                        frame, points[i], points[i + 1],
                        (255, 200, 0), 1)
                if feature_name in (
                    'left_eye', 'right_eye',
                    'top_lip', 'bottom_lip'
                ):
                    if len(points) > 2:
                        cv2.line(
                            frame, points[-1], points[0],
                            (255, 200, 0), 1)
                for pt in points:
                    cv2.circle(frame, pt, 1, (255, 220, 50), -1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   STABILITY TRACKING & ATTENDANCE TRIGGER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _update_stability(self, faces):
        """Track face stability and trigger attendance when ready."""
        recognized = [f for f in faces if f.get('worker_id')]

        if recognized:
            # Pick the primary (largest) face
            primary = max(
                recognized,
                key=lambda f: (
                    (f['box'][2] - f['box'][0])
                    * (f['box'][1] - f['box'][3])
                ),
            )
            worker_id = primary['worker_id']
            worker_name = primary.get('name', '')
            worker_code = primary.get('worker_code', '')

            # Cooldown check
            if (worker_id in self.cooldowns
                    and time.time() < self.cooldowns[worker_id]):
                remaining = self.cooldowns[worker_id] - time.time()
                self._update_stability_ui(
                    worker_name, 0.0,
                    f"Cooldown: {int(remaining)}s remaining",
                    TEXT_SEC)
                return

            # Update tracker
            self.stability.update(worker_id, worker_name, worker_code)
            progress = self.stability.progress
            dur = self.stability.duration
            req = Config.STABILITY_SECONDS

            if self.stability.is_stable():
                if self.attendance_triggered_for != worker_id:
                    self.attendance_triggered_for = worker_id
                    self._process_attendance(primary)
                self._update_stability_ui(
                    worker_name, 1.0, "Processing…", GOLD)
            else:
                self._update_stability_ui(
                    worker_name, progress,
                    f"Hold steady: {dur:.1f}s / {req:.1f}s",
                    SUCCESS)
        else:
            now = time.time()
            if (self.stability.is_active
                    and self.stability.last_seen
                    and now - self.stability.last_seen > 1.5):
                self.stability.reset()
                self.attendance_triggered_for = None

            if not self.stability.is_active:
                unknown = [f for f in faces if not f.get('worker_id')]
                if unknown:
                    self._update_stability_ui(
                        "Unknown person", 0.0,
                        "Face not recognized", DANGER)
                else:
                    self._update_stability_ui(
                        "No face detected", 0.0,
                        "Waiting for worker…", TEXT_SEC)

    def _update_stability_ui(
        self, name: str, progress: float,
        text: str, color: str
    ):
        try:
            self.stab_name_label.config(text=name, fg=color)
            self.stab_bar_fill.config(bg=color)
            self.stab_bar_fill.place(
                x=0, y=0, relheight=1.0, relwidth=progress)
            self.stab_text_label.config(text=text)
        except tk.TclError:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   ATTENDANCE PROCESSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _process_attendance(self, face_info: Dict[str, Any]):
        """Process attendance in background thread."""
        worker_id = face_info['worker_id']
        worker_name = face_info.get('name', '')
        worker_code = face_info.get('worker_code', '')

        logger.info(
            f"Processing attendance for {worker_name} "
            f"(ID: {worker_id})")

        def do_process():
            result = self.attendance_logger.process_attendance(worker_id)

            # Set cooldown
            self.cooldowns[worker_id] = (
                time.time() + Config.COOLDOWN_SECONDS)

            # Format notification
            action = result.get('action', '')
            now_str = datetime.now().strftime('%I:%M %p')

            if result.get('success'):
                if action == 'timein':
                    notif = {
                        'type': 'timein',
                        'title': 'TIME IN RECORDED',
                        'detail': (
                            f"{worker_name} ({worker_code})\n"
                            f"Clocked in at {now_str}"),
                    }
                elif action == 'timeout':
                    hours = result.get('hours_worked', 0)
                    notif = {
                        'type': 'timeout',
                        'title': 'TIME OUT RECORDED',
                        'detail': (
                            f"{worker_name} ({worker_code})\n"
                            f"Clocked out at {now_str}\n"
                            f"Hours worked: {hours:.1f}h"),
                    }
                else:
                    notif = {
                        'type': 'completed',
                        'title': result.get('message', 'Processed'),
                        'detail': worker_name,
                    }
            else:
                ntype = (action
                         if action in STATUS_COLORS
                         else 'error')
                notif = {
                    'type': ntype,
                    'title': result.get('message', 'Error'),
                    'detail': f"{worker_name} ({worker_code})",
                }

            self.root.after(0, lambda: self._show_notification(notif))
            self.root.after(200, self._update_summary)

        threading.Thread(target=do_process, daemon=True).start()

    def _show_notification(self, notif: Dict[str, Any]):
        """Display notification in the status card."""
        self.notification = notif
        self.notification_expiry = (
            time.time() + Config.DISPLAY_FEEDBACK_SECONDS)

        ntype = notif.get('type', 'error')
        colors = STATUS_COLORS.get(ntype, STATUS_COLORS['error'])

        try:
            bg = colors['bg']
            fg = colors['fg']

            self.notif_card.config(bg=bg)
            for child in self.notif_card.winfo_children():
                try:
                    child.config(bg=bg)
                except tk.TclError:
                    pass

            self.notif_icon_label.config(
                text=colors['icon'], fg=fg, bg=bg)
            self.notif_title_label.config(
                text=notif['title'], fg=fg, bg=bg)
            self.notif_detail_label.config(
                text=notif.get('detail', ''), bg=bg)
        except tk.TclError:
            pass

    def _update_summary(self):
        """Refresh today's attendance counts."""
        if not self.mysql_db or not self.mysql_db.is_connected:
            return

        try:
            today_str = date.today().isoformat()

            if Config.PROJECT_ID:
                row = self.mysql_db.fetch_one("""
                    SELECT
                        COUNT(DISTINCT a.worker_id)
                            AS present_count,
                        COUNT(DISTINCT CASE
                            WHEN a.time_out IS NOT NULL
                            THEN a.worker_id END)
                            AS completed_count
                    FROM attendance a
                    JOIN project_workers pw
                        ON a.worker_id = pw.worker_id
                    WHERE a.attendance_date = %s
                    AND a.is_archived = 0
                    AND pw.project_id = %s
                    AND pw.is_active = 1
                """, (today_str, Config.PROJECT_ID))
            else:
                row = self.mysql_db.fetch_one("""
                    SELECT
                        COUNT(DISTINCT worker_id)
                            AS present_count,
                        COUNT(DISTINCT CASE
                            WHEN time_out IS NOT NULL
                            THEN worker_id END)
                            AS completed_count
                    FROM attendance
                    WHERE attendance_date = %s
                    AND is_archived = 0
                """, (today_str,))

            if row:
                self.present_label.config(
                    text=str(row['present_count']))
                self.completed_label.config(
                    text=str(row['completed_count']))
        except Exception as e:
            logger.error(f"Failed to update summary: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   BACKGROUND WORKERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _recognition_worker(self):
        """Background thread — runs face recognition at ~10 fps."""
        logger.info("Recognition worker started")

        while self.is_running:
            if not self.camera:
                time.sleep(0.5)
                continue

            ret, frame = self.camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)

            try:
                faces = self.face_recognizer.detect_and_recognize(frame)
                with self.faces_lock:
                    self.current_faces = faces
            except Exception as e:
                logger.error(f"Recognition error: {e}")

            time.sleep(0.1)  # ~10 recognition fps

        logger.info("Recognition worker stopped")

    def _sync_worker(self):
        """Background sync thread."""
        logger.info("Sync worker started")

        while self.is_running:
            time.sleep(Config.SYNC_INTERVAL_SECONDS)

            try:
                # Try reconnecting if disconnected
                if self.mysql_db and not self.mysql_db.is_connected:
                    if self.mysql_db.connect():
                        logger.info("MySQL reconnected")
                        self.root.after(
                            0,
                            lambda: self.connection_label.config(
                                text="● Online", fg=SUCCESS))
                        # Reload encodings on reconnect
                        try:
                            count = (
                                self.face_recognizer.load_encodings(
                                    project_id=Config.PROJECT_ID))
                            self.encoding_count = count
                            self.root.after(
                                0,
                                lambda c=count:
                                    self.encoding_label.config(
                                        text=f"Faces: {c}"))
                        except Exception:
                            pass

                # Sync pending records
                if self.sync_manager:
                    result = self.sync_manager.sync_all()
                    synced = result.get('synced', 0)
                    pending = result.get('pending', 0)

                    status = (f"Sync: {pending} pending"
                              if pending > 0
                              else "Sync: OK")
                    self.root.after(
                        0,
                        lambda s=status:
                            self.sync_label.config(text=s))
            except Exception as e:
                logger.error(f"Sync error: {e}")

        logger.info("Sync worker stopped")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   UI UPDATES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _update_clock(self):
        now = datetime.now()
        try:
            self.time_label.config(text=now.strftime('%I:%M:%S %p'))
            self.date_label.config(text=now.strftime('%B %d, %Y'))
        except tk.TclError:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   ACTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

    def _handle_escape(self):
        if self.is_fullscreen:
            self._toggle_fullscreen()
        else:
            self._shutdown()

    def _reload_encodings(self):
        logger.info("Reloading face encodings...")

        def reload():
            try:
                count = self.face_recognizer.load_encodings(
                    project_id=Config.PROJECT_ID)
                self.encoding_count = count
                self.root.after(
                    0,
                    lambda: self.encoding_label.config(
                        text=f"Faces: {count}"))
                logger.info(f"Reloaded {count} encodings")
            except Exception as e:
                logger.error(f"Reload failed: {e}")

        threading.Thread(target=reload, daemon=True).start()

    def _shutdown(self):
        logger.info("Shutting down...")
        self.is_running = False

        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass

        if self.mysql_db:
            try:
                self.mysql_db.close()
            except Exception:
                pass

        try:
            self.root.destroy()
        except Exception:
            pass

        logger.info("Shutdown complete")

    def run(self):
        self.root.mainloop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("\n" + "=" * 60)
    print("  TrackSite Attendance System")
    print("  Tkinter GUI — Automatic Time In / Time Out")
    print("=" * 60 + "\n")

    try:
        app = AttendanceApp()
        app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        try:
            messagebox.showerror("Fatal Error", str(e))
        except Exception:
            print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
