"""
TrackSite Attendance System — Tkinter GUI Application

Standalone facial recognition attendance for construction sites.

Features:
  • Live camera feed with face detection & landmarks
  • Automatic Time In / Time Out (no manual toggle)
  • Anti-accidental safeguards (stability check, cooldown, min interval)
  • Offline-first local SQLite with automatic sync
  • Per-project worker filtering
  • Attendance records display with worker details
  • Project selection at startup

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
from datetime import datetime, date, timedelta
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
        self.root.minsize(1280, 700)

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
        
        # Selected project
        self.selected_project_id: Optional[int] = None
        
        # Attendance records cache
        self.attendance_records: List[Dict[str, Any]] = []

        # ── Initialize databases first for project selection ─
        self._init_databases()
        
        # ── Show project selection dialog ─────────────────
        if not self._show_project_selection():
            self.root.destroy()
            return

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

        # Attendance refresh thread
        self.attendance_refresh_thread = threading.Thread(
            target=self._attendance_refresh_worker, daemon=True)
        self.attendance_refresh_thread.start()

        self._camera_loop()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   DATABASE INITIALIZATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _init_databases(self):
        """Initialize database connections."""
        self.mysql_db = MySQLDatabase()
        self.sqlite_db = SQLiteDatabase()

        try:
            self.mysql_db.connect()
        except Exception:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   PROJECT SELECTION DIALOG
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _load_saved_project(self) -> Optional[int]:
        """Load previously saved project ID from local config."""
        try:
            saved = self.sqlite_db.get_device_config('selected_project_id')
            if saved:
                return int(saved)
        except Exception:
            pass
        return None

    def _save_project_selection(self, project_id: int) -> None:
        """Save selected project ID to local config."""
        try:
            self.sqlite_db.set_device_config('selected_project_id', str(project_id))
            logger.info(f"Saved project ID: {project_id}")
        except Exception as e:
            logger.error(f"Failed to save project selection: {e}")

    def _show_project_selection(self, force: bool = False) -> bool:
        """Show project selection dialog. Returns True if project selected."""
        if not self.mysql_db or not self.mysql_db.is_connected:
            # If no MySQL, use config PROJECT_ID or saved project
            saved_project = self._load_saved_project()
            if saved_project:
                self.selected_project_id = saved_project
                return True
            if Config.PROJECT_ID:
                self.selected_project_id = Config.PROJECT_ID
                return True
            messagebox.showerror(
                "Connection Error",
                "Cannot connect to database.\nPlease check your connection.")
            return False

        # Check for saved project (skip dialog if not forced)
        if not force:
            saved_project = self._load_saved_project()
            if saved_project:
                # Verify project still exists and is active
                project = self.mysql_db.fetch_one("""
                    SELECT project_id FROM projects 
                    WHERE project_id = %s AND is_archived = 0 AND status = 'active'
                """, (saved_project,))
                if project:
                    self.selected_project_id = saved_project
                    return True

        # Fetch active projects
        projects = self.mysql_db.fetch_all("""
            SELECT project_id, project_name, location, status,
                   (SELECT COUNT(*) FROM project_workers pw 
                    WHERE pw.project_id = p.project_id AND pw.is_active = 1) as worker_count
            FROM projects p
            WHERE is_archived = 0 AND status = 'active'
            ORDER BY project_name
        """)

        if not projects:
            messagebox.showerror(
                "No Projects",
                "No active projects found.\nPlease create a project first.")
            return False

        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Project")
        dialog.configure(bg=BG)
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 500) // 2
        y = (dialog.winfo_screenheight() - 400) // 2
        dialog.geometry(f"500x400+{x}+{y}")

        selected_id = tk.IntVar(value=0)
        result = {'selected': False}

        # Header
        header = tk.Frame(dialog, bg=HEADER_BG, height=60)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Frame(header, bg=GOLD, width=4, height=30).pack(side='left', padx=(20, 10), pady=15)
        tk.Label(header, text="TrackSite", font=(FONT, 16, 'bold'),
                 fg=GOLD, bg=HEADER_BG).pack(side='left', pady=15)
        tk.Label(header, text="Select Project", font=(FONT, 14),
                 fg=TEXT_SEC, bg=HEADER_BG).pack(side='left', padx=(10, 0), pady=15)

        # Content
        content = tk.Frame(dialog, bg=BG)
        content.pack(fill='both', expand=True, padx=20, pady=15)

        tk.Label(content, text="Choose a project to use this device for:",
                 font=(FONT, 11), fg=TEXT_SEC, bg=BG).pack(anchor='w', pady=(0, 10))

        # Project list with scrollbar
        list_frame = tk.Frame(content, bg=CARD_BG, highlightthickness=1,
                              highlightbackground=SEPARATOR)
        list_frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(list_frame, bg=CARD_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=CARD_BG)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Make scrollable frame expand to canvas width
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(
            canvas_window, width=e.width))

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Store radio buttons to update their selection visually
        radio_buttons = []

        for project in projects:
            proj_frame = tk.Frame(scrollable_frame, bg=CARD_BG, cursor='hand2')
            proj_frame.pack(fill='x', padx=10, pady=5)

            rb = tk.Radiobutton(
                proj_frame, text='', variable=selected_id,
                value=project['project_id'], bg=CARD_BG,
                activebackground=CARD_BG, selectcolor=GOLD,
                highlightthickness=0, bd=0)
            rb.pack(side='left', padx=(5, 10))
            radio_buttons.append(rb)

            info_frame = tk.Frame(proj_frame, bg=CARD_BG)
            info_frame.pack(side='left', fill='x', expand=True, pady=8)

            tk.Label(info_frame, text=project['project_name'],
                     font=(FONT, 12, 'bold'), fg=TEXT, bg=CARD_BG,
                     anchor='w').pack(fill='x')
            
            location = project.get('location', 'No location')
            if location and len(location) > 50:
                location = location[:47] + '...'
            
            tk.Label(info_frame, text=f"📍 {location}",
                     font=(FONT, 9), fg=TEXT_SEC, bg=CARD_BG,
                     anchor='w').pack(fill='x')
            
            tk.Label(info_frame, text=f"👷 {project['worker_count']} workers assigned",
                     font=(FONT, 9), fg=TEXT_SEC, bg=CARD_BG,
                     anchor='w').pack(fill='x')

            # Make entire frame clickable - use invoke() on radio button
            def make_click_handler(radio_btn):
                def handler(event):
                    radio_btn.invoke()
                return handler

            click_handler = make_click_handler(rb)
            proj_frame.bind('<Button-1>', click_handler)
            info_frame.bind('<Button-1>', click_handler)
            for child in info_frame.winfo_children():
                child.bind('<Button-1>', click_handler)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Buttons
        btn_frame = tk.Frame(dialog, bg=BG)
        btn_frame.pack(fill='x', padx=20, pady=15)

        def on_cancel():
            canvas.unbind_all("<MouseWheel>")
            dialog.destroy()

        def on_select():
            if selected_id.get() > 0:
                self.selected_project_id = selected_id.get()
                self._save_project_selection(self.selected_project_id)
                result['selected'] = True
                canvas.unbind_all("<MouseWheel>")
                dialog.destroy()
            else:
                messagebox.showwarning("Select Project", "Please select a project.")

        tk.Button(btn_frame, text="Cancel", font=(FONT, 10),
                  bg=SEPARATOR, fg=TEXT, width=12, cursor='hand2',
                  command=on_cancel).pack(side='left')

        select_btn = tk.Button(btn_frame, text="Select Project", font=(FONT, 10, 'bold'),
                  bg=GOLD, fg=HEADER_BG, width=15, cursor='hand2',
                  command=on_select)
        select_btn.pack(side='right')

        # Bind Enter key to select
        dialog.bind('<Return>', lambda e: on_select())
        dialog.bind('<KP_Enter>', lambda e: on_select())
        
        # Focus the dialog so it receives key events
        dialog.focus_set()

        # Handle window close button
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Wait for dialog
        self.root.wait_window(dialog)
        
        return result['selected']

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

        # Change Project button
        self.change_project_btn = tk.Button(
            left, text="⚙ Change", font=(FONT, 9),
            bg=SEPARATOR, fg=TEXT, cursor='hand2',
            command=self._change_project, relief='flat', padx=8)
        self.change_project_btn.pack(side='left', padx=(12, 0))

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
    #   MAIN AREA (Attendance Table + Camera/Status Panel)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_main_area(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=12, pady=(8, 4))

        # Create horizontal paned layout
        # Left: Attendance Records (main focus - larger)
        # Right: Camera + Detection Status (smaller - about 1/3)

        # Left section: Attendance Records Table (main focus)
        left_section = tk.Frame(main, bg=BG)
        left_section.pack(side='left', fill='both', expand=True, padx=(0, 8))

        self._build_attendance_table(left_section)

        # Right section: Camera + Detection Status
        right_section = tk.Frame(main, bg=BG, width=400)
        right_section.pack(side='right', fill='y')
        right_section.pack_propagate(False)

        self._build_camera_panel(right_section)

    def _build_camera_panel(self, parent):
        """Build the camera and detection status panel (right side)."""

        # ── Camera Feed ───────────────────────────────────
        camera_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        camera_card.pack(fill='both', expand=True, pady=(0, 8))

        tk.Label(
            camera_card, text="CAMERA FEED",
            font=(FONT, 9, 'bold'), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w').pack(fill='x', padx=10, pady=(8, 4))

        camera_container = tk.Frame(camera_card, bg=CAMERA_BG, height=280)
        camera_container.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        camera_container.pack_propagate(False)

        self.camera_label = tk.Label(camera_container, bg=CAMERA_BG)
        self.camera_label.pack(fill='both', expand=True)

        self.camera_placeholder = tk.Label(
            camera_container,
            text="📷\nInitializing...",
            font=(FONT, 14), fg=TEXT_SEC, bg=CAMERA_BG,
            justify='center')
        self.camera_placeholder.place(relx=0.5, rely=0.5, anchor='center')

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
            anchor='w', wraplength=370, justify='left')
        self.stab_name_label.pack(fill='x', padx=14)

        # Progress bar
        bar_container = tk.Frame(stab_card, bg=SEPARATOR, height=10)
        bar_container.pack(fill='x', padx=14, pady=(8, 4))
        bar_container.pack_propagate(False)

        self.stab_bar_fill = tk.Frame(
            bar_container, bg=TEXT_SEC, height=10)
        self.stab_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self.stab_text_label = tk.Label(
            stab_card, text="Waiting for worker...",
            font=(FONT, 10), fg=TEXT_SEC, bg=CARD_BG,
            anchor='w', wraplength=370, justify='left')
        self.stab_text_label.pack(fill='x', padx=14, pady=(0, 12))

        # ── Quick Stats ──────────────────────────────────
        stats_card = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=SEPARATOR, highlightthickness=1)
        stats_card.pack(fill='x')

        stats_frame = tk.Frame(stats_card, bg=CARD_BG)
        stats_frame.pack(fill='x', padx=14, pady=12)

        # Present count
        present_frame = tk.Frame(stats_frame, bg=CARD_BG)
        present_frame.pack(side='left', expand=True)
        self.present_label = tk.Label(
            present_frame, text="0", font=(FONT, 28, 'bold'),
            fg=SUCCESS, bg=CARD_BG)
        self.present_label.pack()
        tk.Label(present_frame, text="Present",
                 font=(FONT, 9), fg=TEXT_SEC, bg=CARD_BG).pack()

        # Completed count
        completed_frame = tk.Frame(stats_frame, bg=CARD_BG)
        completed_frame.pack(side='left', expand=True)
        self.completed_label = tk.Label(
            completed_frame, text="0", font=(FONT, 28, 'bold'),
            fg=PRIMARY, bg=CARD_BG)
        self.completed_label.pack()
        tk.Label(completed_frame, text="Completed",
                 font=(FONT, 9), fg=TEXT_SEC, bg=CARD_BG).pack()

    def _build_info_panel(self, parent):
        """Legacy - kept for compatibility but not used."""
        pass

    def _build_attendance_table(self, parent):
        """Build the attendance records table section (main focus)."""
        # Section header
        table_header = tk.Frame(parent, bg=BG)
        table_header.pack(fill='x', pady=(0, 8))

        tk.Label(table_header, text="TODAY'S ATTENDANCE RECORDS",
                 font=(FONT, 12, 'bold'), fg=GOLD, bg=BG).pack(side='left')

        self.record_count_label = tk.Label(
            table_header, text="0 records",
            font=(FONT, 10), fg=TEXT_SEC, bg=BG)
        self.record_count_label.pack(side='right')

        # Table container with border - main focus element
        table_container = tk.Frame(
            parent, bg=CARD_BG,
            highlightbackground=GOLD, highlightthickness=2)
        table_container.pack(fill='both', expand=True, pady=(0, 5))

        # Create Treeview with scrollbar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Attendance.Treeview',
                        background=CARD_BG,
                        foreground=TEXT,
                        fieldbackground=CARD_BG,
                        rowheight=32,
                        font=(FONT, 10))
        style.configure('Attendance.Treeview.Heading',
                        background=HEADER_BG,
                        foreground=TEXT,
                        font=(FONT, 10, 'bold'),
                        padding=(8, 6))
        style.map('Attendance.Treeview',
                  background=[('selected', GOLD)],
                  foreground=[('selected', HEADER_BG)])

        columns = ('worker_name', 'worker_code', 'time_in', 'time_out', 
                   'status', 'hours', 'role', 'classification', 'schedule')
        
        tree_frame = tk.Frame(table_container, bg=CARD_BG)
        tree_frame.pack(fill='both', expand=True, padx=2, pady=2)

        self.attendance_tree = ttk.Treeview(
            tree_frame, columns=columns, show='headings',
            style='Attendance.Treeview', height=15)

        # Define columns
        self.attendance_tree.heading('worker_name', text='Worker Name')
        self.attendance_tree.heading('worker_code', text='Code')
        self.attendance_tree.heading('time_in', text='Time In')
        self.attendance_tree.heading('time_out', text='Time Out')
        self.attendance_tree.heading('status', text='Status')
        self.attendance_tree.heading('hours', text='Hours')
        self.attendance_tree.heading('role', text='Role')
        self.attendance_tree.heading('classification', text='Classification')
        self.attendance_tree.heading('schedule', text='Schedule')

        # Column widths
        self.attendance_tree.column('worker_name', width=150, minwidth=120)
        self.attendance_tree.column('worker_code', width=80, minwidth=60)
        self.attendance_tree.column('time_in', width=80, minwidth=70)
        self.attendance_tree.column('time_out', width=80, minwidth=70)
        self.attendance_tree.column('status', width=80, minwidth=60)
        self.attendance_tree.column('hours', width=60, minwidth=50)
        self.attendance_tree.column('role', width=100, minwidth=80)
        self.attendance_tree.column('classification', width=100, minwidth=80)
        self.attendance_tree.column('schedule', width=100, minwidth=80)

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical',
                                   command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)

        self.attendance_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

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

        # Database already initialized in _init_databases

        if self.mysql_db and self.mysql_db.is_connected:
            self.connection_label.config(text="● Online", fg=SUCCESS)
            logger.info("MySQL connected")
        else:
            self.connection_label.config(text="● Offline", fg=WARNING)
            logger.warning("MySQL unavailable — running offline")

        # Load project info using selected project
        self._load_project_info()

        # Core components
        self.face_recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
        self.attendance_logger = AttendanceLogger(
            self.mysql_db, self.sqlite_db)
        self.sync_manager = SyncManager(self.mysql_db, self.sqlite_db)

        # Load face encodings for selected project
        try:
            self.encoding_count = self.face_recognizer.load_encodings(
                project_id=self.selected_project_id)
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

        # Initial summary and attendance records
        self._update_summary()
        self._refresh_attendance_records()

        logger.info("Initialization complete")

    def _load_project_info(self):
        """Load project name from database."""
        project_id = self.selected_project_id or Config.PROJECT_ID
        
        if project_id and self.mysql_db and self.mysql_db.is_connected:
            project = self.mysql_db.fetch_one(
                "SELECT project_name FROM projects WHERE project_id = %s",
                (project_id,))
            if project:
                self.project_name = project['project_name']
                self.project_label.config(
                    text=f"  │  {self.project_name}")
                return

        self.project_label.config(text="  Attendance System")

    def _refresh_attendance_records(self):
        """Refresh the attendance records table with today's data."""
        if not self.mysql_db or not self.mysql_db.is_connected:
            return

        project_id = self.selected_project_id or Config.PROJECT_ID
        today_str = date.today().isoformat()
        day_name = date.today().strftime('%A').lower()

        try:
            if project_id:
                records = self.mysql_db.fetch_all("""
                    SELECT 
                        CONCAT(w.first_name, ' ', w.last_name) as worker_name,
                        w.worker_code,
                        a.time_in,
                        a.time_out,
                        a.status,
                        COALESCE(a.hours_worked, 0) as hours_worked,
                        COALESCE(wt.work_type_name, w.position, 'N/A') as role,
                        COALESCE(wc.classification_name, 'N/A') as classification,
                        COALESCE(ds.start_time, s.start_time) as start_time,
                        COALESCE(ds.end_time, s.end_time) as end_time
                    FROM attendance a
                    JOIN workers w ON a.worker_id = w.worker_id
                    JOIN project_workers pw ON w.worker_id = pw.worker_id
                    LEFT JOIN work_types wt ON w.work_type_id = wt.work_type_id
                    LEFT JOIN worker_classifications wc ON wt.classification_id = wc.classification_id
                    LEFT JOIN daily_schedules ds ON w.worker_id = ds.worker_id 
                        AND ds.schedule_date = %s AND ds.is_active = 1 AND ds.is_rest_day = 0
                    LEFT JOIN schedules s ON w.worker_id = s.worker_id 
                        AND s.day_of_week = %s AND s.is_active = 1
                        AND ds.daily_schedule_id IS NULL
                    WHERE a.attendance_date = %s
                    AND a.is_archived = 0
                    AND pw.project_id = %s
                    AND pw.is_active = 1
                    GROUP BY a.attendance_id
                    ORDER BY a.created_at DESC
                """, (today_str, day_name, today_str, project_id))
            else:
                records = self.mysql_db.fetch_all("""
                    SELECT 
                        CONCAT(w.first_name, ' ', w.last_name) as worker_name,
                        w.worker_code,
                        a.time_in,
                        a.time_out,
                        a.status,
                        COALESCE(a.hours_worked, 0) as hours_worked,
                        COALESCE(wt.work_type_name, w.position, 'N/A') as role,
                        COALESCE(wc.classification_name, 'N/A') as classification,
                        COALESCE(ds.start_time, s.start_time) as start_time,
                        COALESCE(ds.end_time, s.end_time) as end_time
                    FROM attendance a
                    JOIN workers w ON a.worker_id = w.worker_id
                    LEFT JOIN work_types wt ON w.work_type_id = wt.work_type_id
                    LEFT JOIN worker_classifications wc ON wt.classification_id = wc.classification_id
                    LEFT JOIN daily_schedules ds ON w.worker_id = ds.worker_id 
                        AND ds.schedule_date = %s AND ds.is_active = 1 AND ds.is_rest_day = 0
                    LEFT JOIN schedules s ON w.worker_id = s.worker_id 
                        AND s.day_of_week = %s AND s.is_active = 1
                        AND ds.daily_schedule_id IS NULL
                    WHERE a.attendance_date = %s
                    AND a.is_archived = 0
                    GROUP BY a.attendance_id
                    ORDER BY a.created_at DESC
                """, (today_str, day_name, today_str))

            self.attendance_records = records if records else []
            self._update_attendance_table()

        except Exception as e:
            logger.error(f"Failed to refresh attendance records: {e}")

    def _update_attendance_table(self):
        """Update the treeview with current attendance records."""
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        # Add records
        for record in self.attendance_records:
            status = record.get('status', 'present') or 'present'
            status_display = status.replace('_', ' ').title()
            
            hours = record.get('hours_worked', 0) or 0
            hours_display = f"{float(hours):.1f}h" if float(hours) > 0 else '-'
            
            # Helper to format time values (timedelta or datetime.time) to "HH:MM AM/PM"
            def format_time(val):
                if val is None:
                    return '-'
                try:
                    if isinstance(val, timedelta):
                        total_seconds = int(val.total_seconds())
                        hours_val = (total_seconds // 3600) % 24
                        minutes_val = (total_seconds % 3600) // 60
                    elif hasattr(val, 'hour'):
                        hours_val = val.hour
                        minutes_val = val.minute
                    else:
                        return str(val)
                    
                    period = 'AM' if hours_val < 12 else 'PM'
                    display_hour = hours_val % 12
                    if display_hour == 0:
                        display_hour = 12
                    return f"{display_hour}:{minutes_val:02d} {period}"
                except:
                    return str(val) if val else '-'
            
            time_in = format_time(record.get('time_in'))
            time_out = format_time(record.get('time_out'))
            
            # Format schedule (start_time - end_time)
            start_time = record.get('start_time')
            end_time = record.get('end_time')
            if start_time and end_time:
                schedule = f"{format_time(start_time)} - {format_time(end_time)}"
            else:
                schedule = 'Not set'
            
            role = record.get('role') or 'N/A'
            classification = record.get('classification') or 'N/A'
            
            self.attendance_tree.insert('', 'end', values=(
                record.get('worker_name', ''),
                record.get('worker_code', ''),
                time_in,
                time_out,
                status_display,
                hours_display,
                role,
                classification,
                schedule
            ))

        # Update count label
        count = len(self.attendance_records)
        self.record_count_label.config(text=f"{count} record{'s' if count != 1 else ''}")

    def _attendance_refresh_worker(self):
        """Background thread to periodically refresh attendance records."""
        logger.info("Attendance refresh worker started")
        
        while self.is_running:
            time.sleep(30)  # Refresh every 30 seconds
            
            try:
                self.root.after(0, self._refresh_attendance_records)
            except Exception as e:
                logger.error(f"Attendance refresh error: {e}")
        
        logger.info("Attendance refresh worker stopped")

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
        # Don't overwrite active notification (e.g. TIME IN / TIME OUT result)
        if time.time() < self.notification_expiry:
            return
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
                        'detail': f"{worker_name} ({worker_code}) - In: {now_str}",
                    }
                elif action == 'timeout':
                    hours = result.get('hours_worked', 0)
                    notif = {
                        'type': 'timeout',
                        'title': 'TIME OUT RECORDED',
                        'detail': f"{worker_name} ({worker_code}) - Out: {now_str} ({hours:.1f}h)",
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
            self.root.after(500, self._refresh_attendance_records)

        threading.Thread(target=do_process, daemon=True).start()

    def _show_notification(self, notif: Dict[str, Any]):
        """Display notification feedback in the detection status area."""
        self.notification = notif
        self.notification_expiry = (
            time.time() + Config.DISPLAY_FEEDBACK_SECONDS)

        ntype = notif.get('type', 'error')
        colors = STATUS_COLORS.get(ntype, STATUS_COLORS['error'])

        try:
            fg = colors['fg']
            icon = colors['icon']

            # Update the detection status to show feedback
            self.stab_name_label.config(
                text=f"{icon} {notif['title']}", fg=fg)
            self.stab_text_label.config(
                text=notif.get('detail', ''))
            self.stab_bar_fill.config(bg=fg)
            self.stab_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=1.0)
        except tk.TclError:
            pass

    def _update_summary(self):
        """Refresh today's attendance counts."""
        if not self.mysql_db or not self.mysql_db.is_connected:
            return

        project_id = self.selected_project_id or Config.PROJECT_ID

        try:
            today_str = date.today().isoformat()

            if project_id:
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
                """, (today_str, project_id))
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

        project_id = self.selected_project_id or Config.PROJECT_ID

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
                                    project_id=project_id))
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
        
        project_id = self.selected_project_id or Config.PROJECT_ID

        def reload():
            try:
                count = self.face_recognizer.load_encodings(
                    project_id=project_id)
                self.encoding_count = count
                self.root.after(
                    0,
                    lambda: self.encoding_label.config(
                        text=f"Faces: {count}"))
                logger.info(f"Reloaded {count} encodings")
            except Exception as e:
                logger.error(f"Reload failed: {e}")

        threading.Thread(target=reload, daemon=True).start()

    def _change_project(self):
        """Open project selection dialog to change the current project."""
        if not self.mysql_db or not self.mysql_db.is_connected:
            messagebox.showerror(
                "Connection Error",
                "Cannot connect to database.\nPlease check your connection.")
            return

        # Show project selection dialog (forced)
        if self._show_project_selection(force=True):
            # Reload project info and encodings
            self._load_project_info()
            self._reload_encodings()
            self._refresh_attendance_records()
            self._update_summary()
            logger.info(f"Switched to project ID: {self.selected_project_id}")

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
