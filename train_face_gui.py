"""
TrackSite Face Registration — Modern GUI Application

Clean, modern interface (iOS-inspired) for registering worker faces 
for biometric attendance. Replaces the terminal-based train_face.py.

Usage: python train_face_gui.py
       pythonw train_face_gui.py  (no console window)
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import sys
import os
import json
import time

# Ensure correct working directory (for .env loading)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import face_recognition
except ImportError:
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        messagebox.showerror("Missing Dependency",
            "face_recognition is not installed.\n\nRun: pip install face-recognition")
        _root.destroy()
    except:
        print("ERROR: face_recognition not installed.")
    sys.exit(1)

from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Design Tokens — iOS / TrackSite inspired
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG          = '#F0F2F5'        # Light gray background
CARD        = '#FFFFFF'        # White card
HEADER_BG   = '#1A1A1A'       # Dark header
GOLD        = '#DAA520'        # TrackSite gold
DARK_GOLD   = '#B8860B'       # Pressed gold
SUCCESS     = '#34C759'        # iOS green
SUCCESS_BG  = '#E8F5E9'
DANGER      = '#FF3B30'        # iOS red
DANGER_BG   = '#FFEBEE'
WARNING     = '#FF9500'        # iOS orange
WARNING_BG  = '#FFF3E0'
PRIMARY     = '#007AFF'        # iOS blue
PRIMARY_BG  = '#E3F2FD'
TEXT        = '#1C1C1E'        # Primary text
TEXT_SEC    = '#8E8E93'        # Secondary text
SEPARATOR   = '#E5E5EA'       # Borders & dividers
SELECTED    = '#FFF8E1'       # Selected row
HOVER       = '#F9F9F9'       # Hover row
FONT        = 'Segoe UI'      # Windows system font

# Position prompts for guided capture
POSITION_GUIDES = [
    "Look FORWARD (center)",
    "Turn slightly LEFT",
    "Turn slightly RIGHT",
    "Tilt slightly UP",
    "Tilt slightly DOWN",
    "Look FORWARD again",
    "Slight LEFT + UP",
    "Slight RIGHT + UP",
    "Slight LEFT + DOWN",
    "Look FORWARD (final)",
]

NUM_CAPTURES = 10

# Auto-capture settings
AUTO_CAPTURE_HOLD_SECONDS = 1.5   # Hold still with face detected to auto-capture
MIN_FACE_SIZE_RATIO = 0.08        # Face must be at least 8% of frame area
CAPTURE_COOLDOWN = 1.0            # Seconds between auto-captures


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FaceRegistrationApp:
    """Modern GUI for biometric face registration."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("TrackSite — Face Registration")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BG)

        # ── State ──────────────────────────────────
        self.mysql_db = None
        self.sqlite_db = None
        self.workers = []
        self.filtered_workers = []
        self.selected_worker = None
        self.worker_frames = {}

        # Camera state
        self.cap = None
        self.camera_running = False
        self.capturing = False
        self.captured_images = []
        self.face_locations = []
        self.face_landmarks = []
        self.frame_counter = 0
        self.current_raw_frame = None
        self.photo_image = None          # prevent GC
        self.search_entry = None         # reference for focus check
        self.capture_flash_until = 0     # timestamp for flash overlay
        self.face_quality_ok = False     # current frame quality flag

        # ── Build UI ───────────────────────────────
        self._build_header()
        self._build_main()
        self._build_status_bar()

        # ── Database ───────────────────────────────
        self._connect_db()
        self._load_workers()

        # ── Center on screen ───────────────────────
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - 1200) // 2)
        y = max(0, (sh - 750) // 2)
        self.root.geometry(f"1200x750+{x}+{y}")

        # ── Key bindings ──────────────────────────
        self.root.bind('<space>', self._on_space)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   HEADER BAR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_header(self):
        header = tk.Frame(self.root, bg=HEADER_BG, height=56)
        header.pack(fill='x')
        header.pack_propagate(False)

        # Left — branding
        left = tk.Frame(header, bg=HEADER_BG)
        left.pack(side='left', padx=20)

        # Gold accent bar
        tk.Frame(left, bg=GOLD, width=4, height=28).pack(side='left', padx=(0, 12))
        tk.Label(left, text="TrackSite", font=(FONT, 15, 'bold'),
                 fg=GOLD, bg=HEADER_BG).pack(side='left')
        tk.Label(left, text="  Face Registration", font=(FONT, 13),
                 fg='#AAAAAA', bg=HEADER_BG).pack(side='left')

        # Right — close button
        close_btn = tk.Label(header, text="✕", font=(FONT, 15),
                             fg='#666', bg=HEADER_BG, cursor='hand2', padx=16)
        close_btn.pack(side='right')
        close_btn.bind('<Button-1>', lambda e: self._on_close())
        close_btn.bind('<Enter>', lambda e: close_btn.config(fg=DANGER))
        close_btn.bind('<Leave>', lambda e: close_btn.config(fg='#666'))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   MAIN CONTENT (Left list + Right panel)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=20, pady=(15, 8))

        # -- Left panel: worker list (fixed 340px) --
        self._build_left_panel(main)

        # -- Right panel: dynamic content --
        self.right_panel = tk.Frame(main, bg=BG)
        self.right_panel.pack(side='left', fill='both', expand=True, padx=(15, 0))

        self._show_welcome()

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=CARD, width=340, relief='flat',
                        highlightbackground=SEPARATOR, highlightthickness=1)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)

        # Title row
        title_frame = tk.Frame(left, bg=CARD)
        title_frame.pack(fill='x', padx=18, pady=(16, 0))

        tk.Label(title_frame, text="Workers", font=(FONT, 15, 'bold'),
                 fg=TEXT, bg=CARD).pack(side='left')

        self.worker_count_label = tk.Label(title_frame, text="0",
                                           font=(FONT, 10), fg=TEXT_SEC, bg=CARD)
        self.worker_count_label.pack(side='right')

        # Search bar
        search_frame = tk.Frame(left, bg='#F0F0F3', relief='flat',
                                highlightbackground=SEPARATOR, highlightthickness=1)
        search_frame.pack(fill='x', padx=14, pady=(12, 8))

        tk.Label(search_frame, text="🔍", font=(FONT, 10),
                 bg='#F0F0F3', fg=TEXT_SEC).pack(side='left', padx=(8, 2))

        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', lambda *_: self._filter_workers())

        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                     font=(FONT, 11), bg='#F0F0F3', fg=TEXT,
                                     relief='flat', border=0,
                                     insertbackground=TEXT)
        self.search_entry.pack(side='left', fill='x', expand=True, padx=4, pady=7)

        # Separator
        tk.Frame(left, bg=SEPARATOR, height=1).pack(fill='x')

        # Scrollable worker list
        list_container = tk.Frame(left, bg=CARD)
        list_container.pack(fill='both', expand=True)

        self.list_canvas = tk.Canvas(list_container, bg=CARD,
                                     highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(list_container, orient='vertical',
                                  command=self.list_canvas.yview)

        self.list_frame = tk.Frame(self.list_canvas, bg=CARD)
        self.list_frame.bind('<Configure>',
            lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all')))

        self.list_canvas.create_window((0, 0), window=self.list_frame,
                                       anchor='nw', tags='list_window')
        self.list_canvas.configure(yscrollcommand=scrollbar.set)
        self.list_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Mouse-wheel only when hovering over the list
        def _bind_wheel(e):
            self.list_canvas.bind_all("<MouseWheel>",
                lambda ev: self.list_canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units"))
        def _unbind_wheel(e):
            self.list_canvas.unbind_all("<MouseWheel>")

        self.list_canvas.bind('<Enter>', _bind_wheel)
        self.list_canvas.bind('<Leave>', _unbind_wheel)

        # Keep canvas inner frame width in sync
        def _on_canvas_cfg(event):
            self.list_canvas.itemconfig('list_window', width=event.width)
        self.list_canvas.bind('<Configure>', _on_canvas_cfg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   STATUS BAR (bottom)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg='#E8E8ED', height=28)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        self.status_label = tk.Label(bar, text="Ready", font=(FONT, 10),
                                     fg=TEXT_SEC, bg='#E8E8ED')
        self.status_label.pack(side='left', padx=15)

        self.db_status_label = tk.Label(bar, text="● Disconnected",
                                        font=(FONT, 10), fg=DANGER, bg='#E8E8ED')
        self.db_status_label.pack(side='right', padx=15)

    def _set_status(self, text, color=TEXT_SEC):
        try:
            self.status_label.config(text=text, fg=color)
        except tk.TclError:
            pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   DATABASE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _connect_db(self):
        self._set_status("Connecting to database...")
        self.mysql_db = MySQLDatabase()
        self.sqlite_db = SQLiteDatabase()

        if self.mysql_db.connect():
            self.db_status_label.config(text="● Connected", fg=SUCCESS)
            self._set_status("Database connected")
        else:
            self.db_status_label.config(text="● Connection Failed", fg=DANGER)
            self._set_status("Database connection failed!", DANGER)
            messagebox.showerror("Database Error",
                "Cannot connect to MySQL.\n\n"
                "Make sure:\n"
                "  1. XAMPP MySQL is running\n"
                "  2. Database 'construction_management' exists\n"
                "  3. .env has correct credentials")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   WORKER LIST
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _load_workers(self):
        if not self.mysql_db or not self.mysql_db.is_connected:
            return

        self.workers = self.mysql_db.fetch_all("""
            SELECT w.worker_id, w.worker_code, w.first_name, w.last_name,
                   w.position, w.employment_status,
                   fe.encoding_id IS NOT NULL as is_registered
            FROM workers w
            LEFT JOIN face_encodings fe ON w.worker_id = fe.worker_id AND fe.is_active = 1
            WHERE w.is_archived = 0
            ORDER BY w.last_name, w.first_name
        """) or []

        self._filter_workers()

    def _filter_workers(self):
        query = self.search_var.get().lower().strip()
        if not query:
            self.filtered_workers = list(self.workers)
        else:
            self.filtered_workers = [
                w for w in self.workers
                if query in w['first_name'].lower()
                or query in w['last_name'].lower()
                or query in w['worker_code'].lower()
                or (w.get('position') and query in w['position'].lower())
            ]
        self._populate_list()

    def _populate_list(self):
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        self.worker_frames = {}

        count = len(self.filtered_workers)
        self.worker_count_label.config(
            text=f"{count} worker{'s' if count != 1 else ''}")

        if not self.filtered_workers:
            empty = tk.Frame(self.list_frame, bg=CARD)
            empty.pack(fill='x', pady=40)
            tk.Label(empty, text="No workers found", font=(FONT, 12),
                     fg=TEXT_SEC, bg=CARD).pack()
            return

        for worker in self.filtered_workers:
            self._create_worker_row(worker)

    def _create_worker_row(self, worker):
        is_reg = bool(worker.get('is_registered'))
        is_sel = (self.selected_worker
                  and self.selected_worker['worker_id'] == worker['worker_id'])
        bg = SELECTED if is_sel else CARD

        row = tk.Frame(self.list_frame, bg=bg, cursor='hand2')
        row.pack(fill='x')

        content = tk.Frame(row, bg=bg)
        content.pack(fill='x', padx=15, pady=10)

        # Left: name + details
        info = tk.Frame(content, bg=bg)
        info.pack(side='left', fill='x', expand=True)

        name = f"{worker['first_name']} {worker['last_name']}"
        name_lbl = tk.Label(info, text=name, font=(FONT, 12, 'bold'),
                            fg=TEXT, bg=bg, anchor='w')
        name_lbl.pack(anchor='w')

        sub = worker['worker_code']
        if worker.get('position'):
            sub += f"  •  {worker['position']}"
        sub_lbl = tk.Label(info, text=sub, font=(FONT, 9),
                           fg=TEXT_SEC, bg=bg, anchor='w')
        sub_lbl.pack(anchor='w')

        # Right: status pill
        if is_reg:
            status_lbl = tk.Label(content, text="✓ Registered", font=(FONT, 9, 'bold'),
                                  fg=SUCCESS, bg=bg)
        else:
            status_lbl = tk.Label(content, text="○ Unregistered", font=(FONT, 9),
                                  fg=WARNING, bg=bg)
        status_lbl.pack(side='right', padx=(10, 0))

        # Separator line
        tk.Frame(row, bg=SEPARATOR, height=1).pack(fill='x')

        # Click handler
        def on_click(event, w=worker, r=row):
            self._select_worker(w, r)

        for widget in [row, content, info, name_lbl, sub_lbl, status_lbl]:
            widget.bind('<Button-1>', on_click)

        # Hover effects
        def on_enter(e, r=row):
            if not (self.selected_worker
                    and self.selected_worker['worker_id'] == worker['worker_id']):
                self._set_row_bg(r, HOVER)

        def on_leave(e, r=row):
            if not (self.selected_worker
                    and self.selected_worker['worker_id'] == worker['worker_id']):
                self._set_row_bg(r, CARD)

        row.bind('<Enter>', on_enter)
        row.bind('<Leave>', on_leave)

        self.worker_frames[worker['worker_id']] = row

    def _set_row_bg(self, row, bg):
        """Recursively set background on a row and its children."""
        row.configure(bg=bg)
        for child in row.winfo_children():
            try:
                child.configure(bg=bg)
            except tk.TclError:
                pass
            for gc in child.winfo_children():
                try:
                    gc.configure(bg=bg)
                except tk.TclError:
                    pass

    def _select_worker(self, worker, row_frame):
        # Deselect previous
        if self.selected_worker:
            old_id = self.selected_worker['worker_id']
            if old_id in self.worker_frames:
                self._set_row_bg(self.worker_frames[old_id], CARD)

        # Stop camera if active
        if self.camera_running:
            self._stop_camera()

        self.selected_worker = worker
        self._set_row_bg(row_frame, SELECTED)
        self._show_worker_detail(worker)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   RIGHT PANEL — STATES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _clear_right(self):
        for w in self.right_panel.winfo_children():
            w.destroy()

    # ------- WELCOME -------
    def _show_welcome(self):
        self._clear_right()
        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        center = tk.Frame(card, bg=CARD)
        center.place(relx=0.5, rely=0.45, anchor='center')

        tk.Label(center, text="📸", font=(FONT, 48), bg=CARD).pack(pady=(0, 8))
        tk.Label(center, text="Face Registration", font=(FONT, 22, 'bold'),
                 fg=TEXT, bg=CARD).pack()
        tk.Label(center, text="Select a worker from the list to begin",
                 font=(FONT, 12), fg=TEXT_SEC, bg=CARD).pack(pady=(6, 0))

        # Step-by-step instructions
        steps_frame = tk.Frame(center, bg=CARD)
        steps_frame.pack(pady=(30, 0))

        steps = [
            ("1", "Select a worker from the list"),
            ("2", "Start the camera and position your face"),
            ("3", "Capture 10 images at different angles"),
            ("4", "System trains and registers the face"),
        ]
        for num, desc in steps:
            row = tk.Frame(steps_frame, bg=CARD)
            row.pack(fill='x', pady=5)
            tk.Label(row, text=f"  {num}  ", font=(FONT, 10, 'bold'),
                     fg=CARD, bg=GOLD).pack(side='left', padx=(0, 12))
            tk.Label(row, text=desc, font=(FONT, 11), fg=TEXT,
                     bg=CARD, anchor='w').pack(side='left')

    # ------- WORKER DETAIL -------
    def _show_worker_detail(self, worker):
        self._clear_right()
        self.captured_images = []

        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        # Worker info
        info_frame = tk.Frame(card, bg=CARD)
        info_frame.pack(fill='x', padx=30, pady=(28, 0))

        name = f"{worker['first_name']} {worker['last_name']}"
        tk.Label(info_frame, text=name, font=(FONT, 22, 'bold'),
                 fg=TEXT, bg=CARD).pack(anchor='w')

        details = worker['worker_code']
        if worker.get('position'):
            details += f"  •  {worker['position']}"
        tk.Label(info_frame, text=details, font=(FONT, 12),
                 fg=TEXT_SEC, bg=CARD).pack(anchor='w', pady=(4, 0))

        # Registration status banner
        is_reg = bool(worker.get('is_registered'))
        if is_reg:
            s_bg, s_fg = SUCCESS_BG, SUCCESS
            s_text = "✓  This worker already has a face registered"
            btn_text = "Re-Register Face"
        else:
            s_bg, s_fg = WARNING_BG, WARNING
            s_text = "○  This worker has no face registered yet"
            btn_text = "Start Registration"

        banner = tk.Frame(card, bg=s_bg)
        banner.pack(fill='x', padx=30, pady=(15, 0))
        tk.Label(banner, text=s_text, font=(FONT, 11),
                 fg=s_fg, bg=s_bg, pady=10, padx=15).pack(anchor='w')

        # Camera preview area — fills available space
        self.camera_container = tk.Frame(card, bg='#111111')
        self.camera_container.pack(fill='both', expand=True, padx=16, pady=(12, 0))
        self.camera_container.pack_propagate(False)

        # Camera label for live preview
        self.camera_label = tk.Label(self.camera_container, bg='#000')
        self.camera_label.pack(fill='both', expand=True)

        # Controls area
        self.controls_frame = tk.Frame(card, bg=CARD)
        self.controls_frame.pack(fill='x', padx=30, pady=(14, 20))
        controls = self.controls_frame

        # Progress row (hidden initially)
        self.progress_frame = tk.Frame(controls, bg=CARD)
        self.guide_label = tk.Label(self.progress_frame, text="",
                                    font=(FONT, 12, 'bold'), fg=GOLD, bg=CARD)
        self.guide_label.pack(side='left')
        self.progress_label = tk.Label(self.progress_frame, text="0 / 10",
                                       font=(FONT, 11), fg=TEXT_SEC, bg=CARD)
        self.progress_label.pack(side='right')

        # Progress bar container
        self.progress_bar_frame = tk.Frame(controls, bg=SEPARATOR, height=6)
        self.progress_bar_fill = tk.Frame(self.progress_bar_frame,
                                          bg=GOLD, width=0, height=6)

        # Buttons row
        btn_row = tk.Frame(controls, bg=CARD)
        btn_row.pack(fill='x')

        self.start_btn = tk.Button(
            btn_row, text=f"📷  {btn_text}",
            font=(FONT, 12, 'bold'), fg=CARD, bg=GOLD,
            activebackground=DARK_GOLD, activeforeground=CARD,
            bd=0, padx=22, pady=10, cursor='hand2', relief='flat',
            command=lambda: self._start_capture(worker))
        self.start_btn.pack(side='left')
        self.start_btn.bind('<Enter>', lambda e: self.start_btn.config(bg=DARK_GOLD))
        self.start_btn.bind('<Leave>', lambda e: self.start_btn.config(bg=GOLD))

        # Capture button (shown after capture starts)
        self.capture_btn = tk.Button(
            btn_row, text="⏺  Capture  (SPACE)",
            font=(FONT, 12, 'bold'), fg=CARD, bg=PRIMARY,
            activebackground='#0056B3', activeforeground=CARD,
            bd=0, padx=22, pady=10, cursor='hand2', relief='flat',
            state='disabled', command=self._capture_image)

        # Cancel button
        self.cancel_btn = tk.Button(
            btn_row, text="Cancel",
            font=(FONT, 11), fg=TEXT_SEC, bg='#E8E8ED',
            activebackground='#D5D5DA', bd=0, padx=16, pady=10,
            cursor='hand2', relief='flat', command=self._cancel_camera)

        self._set_status(f"Selected: {name}")

        # Start camera preview immediately
        self._start_preview()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   CAMERA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _start_preview(self):
        """Open camera and show live preview (before capture begins)."""
        if self.cap and self.cap.isOpened():
            return  # already running

        # Try multiple backends
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap and cap.isOpened():
                    break
                if cap:
                    cap.release()
                    cap = None
            except Exception:
                cap = None

        if not cap or not cap.isOpened():
            # Show placeholder text instead of blocking messagebox
            try:
                self._set_status("Camera not available — connect a camera and try again", DANGER)
                self.camera_label.config(text="📷  No camera detected\nConnect a webcam to preview",
                                         font=(FONT, 14), fg='#666', compound='center')
            except tk.TclError:
                pass
            return

        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_running = True
        self.capturing = False
        self.frame_counter = 0
        self._camera_loop()

    def _start_capture(self, worker):
        """Switch from preview mode to capture mode."""
        # If camera isn't running yet, try to start it first
        if not self.camera_running:
            self._start_preview()
            if not self.camera_running:
                try:
                    self._set_status("Cannot start capture — no camera available", DANGER)
                except tk.TclError:
                    pass
                return

        self.captured_images = []
        self.capturing = True

        # Swap buttons: hide Start, show Capture + Cancel
        self.start_btn.pack_forget()
        self.capture_btn.pack(side='left', padx=(0, 10))
        self.capture_btn.config(state='normal')
        self.cancel_btn.pack(side='left')

        # Show progress row + bar
        self.progress_frame.pack(fill='x', pady=(0, 8))
        self.progress_bar_frame.pack(fill='x', pady=(0, 6))
        self.progress_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)
        self._update_progress()

        self._set_status("Camera active — Press SPACE to capture", SUCCESS)

    def _start_camera(self, worker):
        """Legacy entry — starts preview + capture together."""
        self._start_preview()
        self._start_capture(worker)

    def _check_face_quality(self, face_locations, frame_h, frame_w):
        """Check if the detected face meets quality requirements.
        Returns (is_ok, reason_string)."""
        if len(face_locations) == 0:
            return False, "No face detected"
        if len(face_locations) > 1:
            return False, "Multiple faces — only one person please"

        top, right, bottom, left = face_locations[0]
        face_area = (bottom - top) * (right - left)
        frame_area = frame_h * frame_w
        ratio = face_area / frame_area if frame_area > 0 else 0

        if ratio < MIN_FACE_SIZE_RATIO:
            return False, "Move CLOSER to camera"

        return True, "Good"

    def _camera_loop(self):
        if not self.camera_running:
            return
        if not self.cap or not self.cap.isOpened():
            self.camera_running = False
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(33, self._camera_loop)
            return

        frame = cv2.flip(frame, 1)
        self.current_raw_frame = frame.copy()
        display = frame.copy()
        h, w = display.shape[:2]

        # Face detection (every 3rd frame for performance)
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = 0.25
            small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
            locs = face_recognition.face_locations(small, model='hog')
            self.face_locations = [
                (int(t / scale), int(r / scale),
                 int(b / scale), int(l / scale))
                for t, r, b, l in locs
            ]
            # Get facial landmarks for face mesh overlay
            try:
                self.face_landmarks = face_recognition.face_landmarks(rgb)
            except Exception:
                self.face_landmarks = []

        face_detected = len(self.face_locations) > 0
        self.face_quality_ok, quality_msg = self._check_face_quality(
            self.face_locations, h, w)

        now = time.time()

        # Draw face rectangles with quality color
        for top, right, bottom, left in self.face_locations:
            if self.face_quality_ok:
                color = (0, 255, 0)   # green = good
            elif face_detected:
                color = (0, 165, 255) # orange = detected but issue
            else:
                color = (0, 0, 255)   # red
            cv2.rectangle(display, (left, top), (right, bottom), color, 3)

        # Draw facial landmarks mesh
        if face_detected and self.face_landmarks:
            try:
                for landmarks in self.face_landmarks:
                    for feature_name, points in landmarks.items():
                        for i in range(len(points) - 1):
                            cv2.line(display, points[i], points[i + 1], (0, 255, 255), 1)
                        if feature_name in ('chin', 'left_eyebrow', 'right_eyebrow',
                                            'nose_bridge', 'left_eye', 'right_eye',
                                            'top_lip', 'bottom_lip'):
                            if len(points) > 2:
                                cv2.line(display, points[-1], points[0], (0, 255, 255), 1)
                        for point in points:
                            cv2.circle(display, point, 1, (0, 200, 255), -1)
            except Exception:
                pass

        # ── Overlay: position guide at bottom ──
        count = len(self.captured_images)
        if self.capturing:
            if count < len(POSITION_GUIDES):
                guide_text = POSITION_GUIDES[count]
            else:
                guide_text = "Look FORWARD"
        else:
            guide_text = "Press Start to begin capture"

        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        text_size = cv2.getTextSize(guide_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display, guide_text, (text_x, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

        # ── Overlay: counter top-right (only during capture) ──
        if self.capturing:
            counter_text = f"{count}/{NUM_CAPTURES}"
            cv2.putText(display, counter_text, (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # ── Overlay: status top-left ──
        if self.face_quality_ok:
            cv2.putText(display, "FACE OK", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.capturing:
                cv2.putText(display, "Press SPACE to capture", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        elif face_detected:
            cv2.putText(display, quality_msg, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(display, "NO FACE - Position yourself", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Face guide oval
        cx, cy = w // 2, h // 2
        cv2.ellipse(display, (cx, cy - 20), (int(w * 0.15), int(h * 0.30)),
                    0, 0, 360, (180, 180, 180), 2)

        # Capture flash effect
        if now < self.capture_flash_until:
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (255, 255, 255), 12)
            cv2.putText(display, "CAPTURED!", ((w - 200) // 2, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Enable/disable capture button (only during capture mode)
        if self.capturing:
            try:
                self.capture_btn.config(state='normal' if face_detected else 'disabled')
            except tk.TclError:
                pass

        # Convert frame for tkinter display — fit to container keeping aspect ratio
        rgb_display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_display)

        try:
            cw = self.camera_container.winfo_width()
            ch = self.camera_container.winfo_height()
            if cw > 10 and ch > 10:
                # Maintain aspect ratio — fill container
                img_w, img_h = pil_image.size
                scale_w = cw / img_w
                scale_h = ch / img_h
                scale = max(scale_w, scale_h)  # fill (crop edges)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                # Center crop
                left_crop = (new_w - cw) // 2
                top_crop = (new_h - ch) // 2
                pil_image = pil_image.crop((left_crop, top_crop,
                                            left_crop + cw, top_crop + ch))
        except Exception:
            pass

        self.photo_image = ImageTk.PhotoImage(pil_image)
        try:
            self.camera_label.config(image=self.photo_image)
        except tk.TclError:
            return

        self.root.after(33, self._camera_loop)   # ~30 fps

    def _capture_image(self):
        if not self.camera_running or not self.capturing or self.current_raw_frame is None:
            return
        if not self.face_locations:
            return

        self.captured_images.append(self.current_raw_frame.copy())
        self._update_progress()

        count = len(self.captured_images)
        self._set_status(f"Captured {count}/{NUM_CAPTURES}", SUCCESS)

        # Flash effect — overlay on video feed
        self.capture_flash_until = time.time() + 0.25

        if count >= NUM_CAPTURES:
            self._stop_camera()
            self._show_verify()

    def _update_progress(self):
        count = len(self.captured_images)

        # Guide text
        if count < len(POSITION_GUIDES):
            self.guide_label.config(text=POSITION_GUIDES[count])
        else:
            self.guide_label.config(text="Look FORWARD")

        self.progress_label.config(text=f"{count} / {NUM_CAPTURES}")

        pct = count / NUM_CAPTURES
        self.progress_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=pct)

    def _stop_camera(self):
        self.camera_running = False
        self.capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.face_locations = []
        self.face_landmarks = []
        self.frame_counter = 0

    def _cancel_camera(self):
        self._stop_camera()
        if self.selected_worker:
            self._show_worker_detail(self.selected_worker)
        self._set_status("Ready")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   VERIFY CAPTURES BEFORE TRAINING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _show_verify(self):
        """Show captured images in a grid for user to verify before training."""
        worker = self.selected_worker
        if not worker:
            return

        self._clear_right()
        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        name = f"{worker['first_name']} {worker['last_name']}"

        # Header
        hdr = tk.Frame(card, bg=CARD)
        hdr.pack(fill='x', padx=24, pady=(20, 0))
        tk.Label(hdr, text="Verify Captured Images", font=(FONT, 18, 'bold'),
                 fg=TEXT, bg=CARD).pack(anchor='w')
        tk.Label(hdr, text=f"{name}  —  {len(self.captured_images)} images captured. "
                           f"Check that all images show the correct face.",
                 font=(FONT, 11), fg=TEXT_SEC, bg=CARD, wraplength=600,
                 justify='left').pack(anchor='w', pady=(4, 0))

        # Scrollable grid of thumbnails
        grid_container = tk.Frame(card, bg=CARD)
        grid_container.pack(fill='both', expand=True, padx=24, pady=(12, 0))

        grid_canvas = tk.Canvas(grid_container, bg=CARD, highlightthickness=0)
        grid_scrollbar = ttk.Scrollbar(grid_container, orient='vertical',
                                       command=grid_canvas.yview)
        grid_inner = tk.Frame(grid_canvas, bg=CARD)
        grid_inner.bind('<Configure>',
            lambda e: grid_canvas.configure(scrollregion=grid_canvas.bbox('all')))
        grid_canvas.create_window((0, 0), window=grid_inner, anchor='nw')
        grid_canvas.configure(yscrollcommand=grid_scrollbar.set)
        grid_canvas.pack(side='left', fill='both', expand=True)
        grid_scrollbar.pack(side='right', fill='y')

        self._verify_thumbs = []  # prevent GC
        cols = 5
        for idx, img in enumerate(self.captured_images):
            r, c = divmod(idx, cols)
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((120, 90), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil)
                self._verify_thumbs.append(tk_img)

                cell = tk.Frame(grid_inner, bg='#333', padx=2, pady=2)
                cell.grid(row=r, column=c, padx=6, pady=6)
                tk.Label(cell, image=tk_img, bg='#000').pack()
                tk.Label(grid_inner, text=f"{idx+1}", font=(FONT, 9),
                         fg=TEXT_SEC, bg=CARD).grid(row=r, column=c, sticky='s',
                                                    pady=(0, 0))
            except Exception:
                pass

        # Buttons
        btn_frame = tk.Frame(card, bg=CARD)
        btn_frame.pack(fill='x', padx=24, pady=(14, 20))

        confirm_btn = tk.Button(
            btn_frame, text="✓  Confirm & Register",
            font=(FONT, 12, 'bold'), fg=CARD, bg=SUCCESS,
            activebackground='#2DA44E', activeforeground=CARD,
            bd=0, padx=22, pady=10, cursor='hand2', relief='flat',
            command=lambda: self._process_training())
        confirm_btn.pack(side='left', padx=(0, 12))
        confirm_btn.bind('<Enter>', lambda e: confirm_btn.config(bg='#2DA44E'))
        confirm_btn.bind('<Leave>', lambda e: confirm_btn.config(bg=SUCCESS))

        retake_btn = tk.Button(
            btn_frame, text="Retake All",
            font=(FONT, 11), fg=TEXT_SEC, bg='#E8E8ED',
            activebackground='#D5D5DA', bd=0, padx=16, pady=10,
            cursor='hand2', relief='flat',
            command=lambda: self._show_worker_detail(worker))
        retake_btn.pack(side='left')

        self._set_status(f"Review captures for {name} — Confirm or Retake", PRIMARY)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   TRAINING / PROCESSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _process_training(self):
        worker = self.selected_worker
        if not worker:
            return

        self._show_processing(worker)

        def train():
            worker_id = worker['worker_id']
            name = f"{worker['first_name']} {worker['last_name']}"
            code = worker['worker_code']

            # Deactivate previous encoding if exists
            existing = self.mysql_db.fetch_one(
                "SELECT encoding_id FROM face_encodings "
                "WHERE worker_id = %s AND is_active = 1",
                (worker_id,))
            if existing:
                self.mysql_db.execute_query(
                    "UPDATE face_encodings SET is_active = 0 WHERE worker_id = %s",
                    (worker_id,))

            recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
            success = recognizer.train_new_face(self.captured_images, worker_id)

            if success:
                # Audit trail
                try:
                    self.mysql_db.execute_query("""
                        INSERT INTO audit_trail
                        (user_id, username, user_level, action_type, module,
                         table_name, record_id, record_identifier, old_values,
                         new_values, changes_summary, ip_address, user_agent,
                         severity, is_sensitive, success)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, (
                        None, 'Biometric System', 'super_admin', 'create',
                        'biometric', 'face_encodings',
                        worker_id, f'{name} ({code})', None,
                        json.dumps({
                            'worker_id': worker_id,
                            'worker_name': name,
                            'images_captured': len(self.captured_images)
                        }),
                        f'Biometric Registration — Face registered for '
                        f'{name} ({code}) with {len(self.captured_images)} images (GUI)',
                        'facial_recognition_system', 'FaceRegistrationGUI',
                        'medium', 0, 1
                    ))
                except Exception as e:
                    print(f"Audit log error: {e}")

                self.root.after(0, lambda: self._show_success(worker))
            else:
                self.root.after(0, lambda: self._show_error(worker))

        threading.Thread(target=train, daemon=True).start()

    # ------- PROCESSING SCREEN -------
    def _show_processing(self, worker):
        self._clear_right()
        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        center = tk.Frame(card, bg=CARD)
        center.place(relx=0.5, rely=0.45, anchor='center')

        name = f"{worker['first_name']} {worker['last_name']}"

        tk.Label(center, text="⏳", font=(FONT, 50), bg=CARD).pack(pady=(0, 12))
        tk.Label(center, text="Processing Images…", font=(FONT, 20, 'bold'),
                 fg=TEXT, bg=CARD).pack()
        tk.Label(center, text=f"Training face recognition for {name}",
                 font=(FONT, 12), fg=TEXT_SEC, bg=CARD).pack(pady=(8, 0))
        tk.Label(center, text=f"{len(self.captured_images)} images captured",
                 font=(FONT, 11), fg=TEXT_SEC, bg=CARD).pack(pady=(4, 0))

        self._processing_dots_label = tk.Label(center, text="",
                                               font=(FONT, 14), fg=GOLD, bg=CARD)
        self._processing_dots_label.pack(pady=(18, 0))
        self._animate_dots()

        self._set_status("Processing face data…", PRIMARY)

    def _animate_dots(self):
        try:
            if not self._processing_dots_label.winfo_exists():
                return
        except (tk.TclError, AttributeError):
            return
        current = self._processing_dots_label.cget('text')
        next_text = current + " ●" if len(current) < 12 else ""
        self._processing_dots_label.config(text=next_text)
        self.root.after(350, self._animate_dots)

    # ------- SUCCESS SCREEN -------
    def _show_success(self, worker):
        self._clear_right()
        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        center = tk.Frame(card, bg=CARD)
        center.place(relx=0.5, rely=0.42, anchor='center')

        name = f"{worker['first_name']} {worker['last_name']}"

        tk.Label(center, text="✅", font=(FONT, 56), bg=CARD).pack(pady=(0, 8))
        tk.Label(center, text="Registration Complete!", font=(FONT, 24, 'bold'),
                 fg=SUCCESS, bg=CARD).pack()
        tk.Label(center, text=name, font=(FONT, 16),
                 fg=TEXT, bg=CARD).pack(pady=(10, 0))
        tk.Label(center,
                 text=f"{worker['worker_code']}  •  "
                      f"{len(self.captured_images)} images processed",
                 font=(FONT, 12), fg=TEXT_SEC, bg=CARD).pack(pady=(4, 0))
        tk.Label(center,
                 text="This worker can now use facial recognition for attendance.",
                 font=(FONT, 11), fg=TEXT_SEC, bg=CARD).pack(pady=(14, 0))

        # Action buttons
        btn_frame = tk.Frame(center, bg=CARD)
        btn_frame.pack(pady=(28, 0))

        another_btn = tk.Button(
            btn_frame, text="Register Another",
            font=(FONT, 12, 'bold'), fg=CARD, bg=GOLD,
            activebackground=DARK_GOLD, bd=0, padx=24, pady=10,
            cursor='hand2', relief='flat',
            command=self._register_another)
        another_btn.pack(side='left', padx=(0, 12))
        another_btn.bind('<Enter>', lambda e: another_btn.config(bg=DARK_GOLD))
        another_btn.bind('<Leave>', lambda e: another_btn.config(bg=GOLD))

        done_btn = tk.Button(
            btn_frame, text="Done",
            font=(FONT, 11), fg=TEXT_SEC, bg='#E8E8ED',
            activebackground='#D5D5DA', bd=0, padx=18, pady=10,
            cursor='hand2', relief='flat', command=self._on_close)
        done_btn.pack(side='left')

        self._set_status(f"✓ Successfully registered {name}", SUCCESS)

        # Refresh worker list to show updated status
        self._load_workers()

    # ------- ERROR SCREEN -------
    def _show_error(self, worker):
        self._clear_right()
        card = tk.Frame(self.right_panel, bg=CARD,
                        highlightbackground=SEPARATOR, highlightthickness=1)
        card.pack(fill='both', expand=True)

        center = tk.Frame(card, bg=CARD)
        center.place(relx=0.5, rely=0.42, anchor='center')

        name = f"{worker['first_name']} {worker['last_name']}"

        tk.Label(center, text="❌", font=(FONT, 56), bg=CARD).pack(pady=(0, 8))
        tk.Label(center, text="Registration Failed", font=(FONT, 24, 'bold'),
                 fg=DANGER, bg=CARD).pack()
        tk.Label(center, text=f"Could not process face for {name}",
                 font=(FONT, 12), fg=TEXT_SEC, bg=CARD).pack(pady=(8, 0))
        tk.Label(center,
                 text="Tips:\n"
                      "  • Ensure good, even lighting\n"
                      "  • Keep your face clearly visible\n"
                      "  • Try different angles between captures",
                 font=(FONT, 11), fg=TEXT_SEC, bg=CARD,
                 justify='left').pack(pady=(18, 0))

        btn_frame = tk.Frame(center, bg=CARD)
        btn_frame.pack(pady=(28, 0))

        retry_btn = tk.Button(
            btn_frame, text="Retry",
            font=(FONT, 12, 'bold'), fg=CARD, bg=WARNING,
            activebackground='#E08600', bd=0, padx=24, pady=10,
            cursor='hand2', relief='flat',
            command=lambda: self._show_worker_detail(worker))
        retry_btn.pack(side='left', padx=(0, 12))
        retry_btn.bind('<Enter>', lambda e: retry_btn.config(bg='#E08600'))
        retry_btn.bind('<Leave>', lambda e: retry_btn.config(bg=WARNING))

        back_btn = tk.Button(
            btn_frame, text="Back",
            font=(FONT, 11), fg=TEXT_SEC, bg='#E8E8ED',
            activebackground='#D5D5DA', bd=0, padx=18, pady=10,
            cursor='hand2', relief='flat', command=self._show_welcome)
        back_btn.pack(side='left')

        self._set_status(f"Registration failed for {name}", DANGER)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #   UTILITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _register_another(self):
        self.selected_worker = None
        self.captured_images = []
        self._show_welcome()
        self._load_workers()

    def _on_space(self, event):
        """Handle spacebar — only capture when camera is in capture mode
        and user is NOT typing in the search box."""
        focused = self.root.focus_get()
        if isinstance(focused, tk.Entry):
            return  # Let the Entry widget handle the space normally
        if self.camera_running and self.capturing:
            self._capture_image()
            return 'break'

    def _on_close(self):
        self.camera_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    try:
        app = FaceRegistrationApp()
        app.run()
        return 0
    except Exception as e:
        try:
            messagebox.showerror("Fatal Error", str(e))
        except:
            print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
