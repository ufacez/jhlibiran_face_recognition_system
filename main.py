import logging
import threading
import time
import sys
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer
from models.attendance_logger import AttendanceLogger
from models.sync_manager import SyncManager
from utils.camera import Camera
from utils.gpio_handler import GPIOHandler
from utils.display import Display

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Main attendance system - optimized for performance"""

    def __init__(self):
        logger.info("Initializing TrackSite Attendance System...")

        # Database connections
        self.mysql_db: Optional[MySQLDatabase] = None
        self.sqlite_db: Optional[SQLiteDatabase] = None

        # Core components
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.attendance_logger: Optional[AttendanceLogger] = None
        self.sync_manager: Optional[SyncManager] = None

        # Hardware interfaces
        self.camera: Optional[Camera] = None
        self.gpio: Optional[GPIOHandler] = None
        self.display: Optional[Display] = None

        # System state
        self.is_running = False
        self.timeout_mode = False
        self.last_recognition_time: Optional[datetime] = None

        # Performance optimization
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.skip_frames = 1  # Process EVERY frame for smooth tracking
        self.frame_counter = 0

        # Threading
        self.sync_thread: Optional[threading.Thread] = None

        # Success display overlay (non-blocking)
        self.success_overlay: Optional[Dict[str, Any]] = None
        self.overlay_lock = threading.Lock()
        self.overlay_end_time: Optional[float] = None

        # Confirmation system
        self.pending_worker: Optional[Dict[str, Any]] = None
        self.waiting_for_confirmation = False
        self.confirmation_timeout = 8.0
        self.confirmation_start_time: Optional[float] = None
        self.last_recognized_worker_id: Optional[int] = None
        self.recognition_cooldown = 3.0

        # Lightweight UI params
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing components...")

        try:
            # Initialize databases
            logger.info("Connecting to databases...")
            self.mysql_db = MySQLDatabase()
            self.sqlite_db = SQLiteDatabase()

            mysql_connected = False
            try:
                mysql_connected = self.mysql_db.connect()
            except Exception:
                mysql_connected = False

            if mysql_connected:
                logger.info("MySQL connected")
            else:
                logger.warning("MySQL unavailable - offline mode")

            logger.info("SQLite database ready")

            # Initialize core components
            logger.info("Initializing core components...")
            self.face_recognizer = FaceRecognizer(self.mysql_db, self.sqlite_db)
            self.attendance_logger = AttendanceLogger(self.mysql_db, self.sqlite_db)
            self.sync_manager = SyncManager(self.mysql_db, self.sqlite_db)
            logger.info("Core components initialized")

            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = Camera()
            if not self.camera.initialize():
                logger.error("Camera initialization failed")
                return False

            self.camera.set_fps(30)
            self.camera.set_resolution(640, 480)
            logger.info("Camera initialized (640x480 @ 30fps)")

            # Setup GPIO
            logger.info("Initializing GPIO...")
            self.gpio = GPIOHandler()
            try:
                self.gpio.add_button_callback(self._handle_timeout_button)
            except Exception:
                pass
            logger.info("GPIO initialized")

            # Create display window
            logger.info("Initializing display...")
            self.display = Display()
            self.display.create_window(fullscreen=False)
            logger.info("Display initialized (windowed)")

            # Load face encodings
            logger.info("Loading face encodings...")
            encoding_count = 0
            try:
                encoding_count = self.face_recognizer.load_encodings()
            except Exception as e:
                logger.warning(f"Failed to load encodings: {e}")

            if encoding_count == 0:
                logger.warning("No face encodings loaded")
            else:
                logger.info(f"Loaded {encoding_count} face encodings")

            logger.info("=" * 60)
            logger.info("System ready!")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            return False

    def run(self):
        """Main loop - optimized for minimal lag"""
        logger.info("Starting attendance system...")
        self.is_running = True

        # Start background sync
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()

        try:
            while self.is_running:
                loop_start = time.time()

                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    logger.error("Failed to read camera frame")
                    time.sleep(0.1)
                    continue

                # Mirror for natural preview
                frame = cv2.flip(frame, 1)
                self.frame_counter += 1

                # Confirmation timeout reset
                if self.waiting_for_confirmation and self.confirmation_start_time:
                    if time.time() - self.confirmation_start_time > self.confirmation_timeout:
                        logger.info("Confirmation timeout - resetting")
                        self.pending_worker = None
                        self.waiting_for_confirmation = False
                        self.confirmation_start_time = None
                        self.last_recognized_worker_id = None

                # Recognition (every N frames) - ALWAYS runs, never stops
                if self.frame_counter % self.skip_frames == 0:
                    try:
                        worker_info, frame, face_box = self.face_recognizer.recognize_face(frame)
                    except Exception:
                        out = self.face_recognizer.recognize_face(frame)
                        if isinstance(out, tuple) and len(out) >= 2:
                            worker_info = out[0]
                            frame = out[1]
                            face_box = None
                        else:
                            worker_info = None
                            face_box = None

                    # Handle recognition for confirmation
                    if worker_info and face_box and not self.waiting_for_confirmation:
                        self._handle_recognition(worker_info, face_box)

                # Show confirmation text (no rectangle box, just overlay text)
                if self.waiting_for_confirmation and self.pending_worker:
                    frame = self._draw_confirmation_text(frame, self.pending_worker)

                # Optimized status bar (removed FPS)
                status = self._get_status_text()
                frame = self.display.add_status_bar(frame, status)

                # Mode indicator
                if self.timeout_mode:
                    frame = self.display.add_overlay(
                        frame,
                        "TIME-OUT MODE",
                        position=(8, 36),
                        color=(0, 165, 255),
                        font_scale=0.7
                    )

                # Draw success banner
                with self.overlay_lock:
                    if self.success_overlay is not None:
                        if time.time() < (self.overlay_end_time or 0):
                            frame = self._draw_success_banner(frame, self.success_overlay)
                        else:
                            self.success_overlay = None
                            self.overlay_end_time = None

                # Display frame
                self.display.show_frame(frame)

                # Handle keyboard
                key = self.display.wait_key(1)
                if key == ord('q') or key == 27:
                    logger.info("Quit key pressed")
                    break
                elif key == ord('t'):
                    self._toggle_timeout_mode()
                elif key == ord('r'):
                    self._reload_encodings()
                elif key == ord('c') or key == ord('C'):
                    if self.waiting_for_confirmation and self.pending_worker:
                        self._confirm_attendance()
                elif key == ord('x') or key == ord('X'):
                    if self.waiting_for_confirmation:
                        self._cancel_confirmation()

                # FPS limiter
                next_frame = loop_start + self.frame_time
                sleep_time = next_frame - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            self.shutdown()

    def _handle_recognition(self, worker_info: Dict[str, Any], face_box: Tuple[int, int, int, int]):
        """Store recognized worker and wait for confirmation"""
        now = datetime.now()
        worker_id = worker_info.get('worker_id')

        # Cooldown check
        if self.last_recognized_worker_id == worker_id:
            if self.last_recognition_time:
                time_diff = (now - self.last_recognition_time).total_seconds()
                if time_diff < self.recognition_cooldown:
                    return
        
        self.last_recognition_time = now
        self.last_recognized_worker_id = worker_id
        
        self.pending_worker = worker_info.copy()
        self.waiting_for_confirmation = True
        self.confirmation_start_time = time.time()

        worker_name = f"{worker_info.get('first_name','')} {worker_info.get('last_name','')}"
        logger.info(f"Recognized (waiting for confirmation): {worker_name}")

    def _confirm_attendance(self):
        """User pressed 'C' - process attendance"""
        if not self.pending_worker:
            return

        logger.info("User confirmed recognition")
        worker_name = f"{self.pending_worker.get('first_name','')} {self.pending_worker.get('last_name','')}"
        worker_id = self.pending_worker.get('worker_id', 0)
        worker_code = self.pending_worker.get('worker_code', 'N/A')

        # Process attendance
        result = self._process_attendance(self.pending_worker)

        # Show result overlay
        self._show_result_overlay(result, worker_name, worker_id, worker_code)

        # Reset confirmation state
        self.pending_worker = None
        self.waiting_for_confirmation = False
        self.confirmation_start_time = None

    def _cancel_confirmation(self):
        """User pressed 'X' - cancel confirmation"""
        logger.info("User cancelled confirmation")
        self.pending_worker = None
        self.waiting_for_confirmation = False
        self.confirmation_start_time = None
        self.last_recognized_worker_id = None

    def _draw_confirmation_text(self, frame, worker_info):
        """Draw simple confirmation text overlay - no rectangle box"""
        h, w = frame.shape[:2]

        first = worker_info.get("first_name") or ""
        last = worker_info.get("last_name") or ""
        name = f"{first} {last}".strip() or "Unknown"
        worker_id = worker_info.get("worker_id", 0)
        worker_code = worker_info.get("worker_code", "N/A")

        # Draw text at top center with shadow for readability
        center_x = w // 2
        
        # Line 1: Worker name
        text1 = name
        (text_w, text_h), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        x1 = center_x - text_w // 2
        y1 = 80
        
        cv2.putText(frame, text1, (x1 + 2, y1 + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
        cv2.putText(frame, text1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Line 2: ID and Code
        text2 = f"ID: {worker_id} | Code: {worker_code}"
        (text_w2, text_h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x2 = center_x - text_w2 // 2
        y2 = 120
        
        cv2.putText(frame, text2, (x2 + 2, y2 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, text2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        
        # Line 3: Instructions
        text3 = "Press C to CONFIRM | Press X to CANCEL"
        (text_w3, text_h3), _ = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x3 = center_x - text_w3 // 2
        y3 = 160
        
        cv2.putText(frame, text3, (x3 + 2, y3 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, text3, (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame

    def _draw_success_banner(self, frame: np.ndarray, overlay_data: Dict[str, Any]) -> np.ndarray:
        """Success banner with time-in information"""
        result = overlay_data.get('result', {})
        worker_name = overlay_data.get('worker_name', '')
        worker_id = overlay_data.get('worker_id', 0)
        worker_code = overlay_data.get('worker_code', 'N/A')
        timestamp = overlay_data.get('timestamp', datetime.now())

        h, w = frame.shape[:2]
        banner_h = 90

        if result.get('success'):
            color = (10, 110, 10)
            title = "RECORDED"
            detail = f"{worker_name} (ID: {worker_id})"
            # Format: "Code: XXX | Time In: 02:30 PM"
            time_str = timestamp.strftime('%I:%M %p')
            detail2 = f"Code: {worker_code} | Time In: {time_str}"
        else:
            color = (120, 30, 30)
            title = "ALREADY IN"
            detail = result.get('message', '')
            detail2 = ""

        # Solid banner
        cv2.rectangle(frame, (0, 0), (w, banner_h), color, -1)

        cv2.putText(frame, title, (12, 28), self.font, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, detail, (12, 54), self.font, 0.6, (230, 230, 230), 1)
        if detail2:
            cv2.putText(frame, detail2, (12, 76), self.font, 0.5, (200, 200, 200), 1)

        return frame

    def _process_attendance(self, worker_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process attendance"""
        worker_id = worker_info['worker_id']

        if self.timeout_mode:
            result = self.attendance_logger.log_timeout(worker_id)
            if result.get('success'):
                self.timeout_mode = False
                try:
                    self.gpio.set_led(False)
                except Exception:
                    pass
        else:
            result = self.attendance_logger.log_timein(worker_id)

        return result

    def _show_result_overlay(self, result: Dict[str, Any], worker_name: str, 
                            worker_id: int, worker_code: str):
        """Set success overlay data"""
        current_time = datetime.now()

        overlay_data = {
            'worker_name': worker_name,
            'worker_id': worker_id,
            'worker_code': worker_code,
            'result': result,
            'timestamp': current_time
        }

        with self.overlay_lock:
            self.success_overlay = overlay_data
            self.overlay_end_time = time.time() + 2.5

    def _toggle_timeout_mode(self):
        """Toggle time-out mode"""
        self.timeout_mode = not self.timeout_mode
        try:
            self.gpio.set_led(self.timeout_mode)
        except Exception:
            pass
        mode_text = "TIME-OUT MODE" if self.timeout_mode else "TIME-IN MODE"
        logger.info(mode_text)

    def _handle_timeout_button(self):
        """GPIO button callback"""
        logger.info("Timeout button pressed")
        self._toggle_timeout_mode()

    def _reload_encodings(self):
        """Reload face encodings"""
        logger.info("Reloading face encodings...")
        try:
            count = self.face_recognizer.load_encodings()
            logger.info(f"Reloaded {count} faces")
        except Exception as e:
            logger.warning(f"Reload failed: {e}")

    def _get_status_text(self) -> str:
        """Get optimized status text - removed FPS, added time and full date"""
        parts = []
        
        # Online/Offline status
        parts.append("[ONLINE]" if (self.mysql_db and getattr(self.mysql_db, 'is_connected', False)) else "[OFFLINE]")
        
        # Mode (Full text instead of abbreviation)
        parts.append("TIME OUT" if self.timeout_mode else "TIME IN")
        
        # Current time in 12-hour format
        now = datetime.now()
        parts.append(now.strftime('%I:%M:%S %p'))
        
        # Full date format (e.g., "November 30, 2025")
        parts.append(now.strftime('%B %d, %Y'))

        return " | ".join(parts)

    def _sync_worker(self):
        """Background sync worker"""
        logger.info("Sync worker started")

        while self.is_running:
            time.sleep(Config.SYNC_INTERVAL_SECONDS)

            try:
                if self.mysql_db and not getattr(self.mysql_db, 'is_connected', False):
                    if self.mysql_db.connect():
                        logger.info("MySQL reconnected")
                        try:
                            self.face_recognizer.load_encodings()
                        except Exception:
                            pass

                if self.sync_manager:
                    result = self.sync_manager.sync_all()
                    if result.get('synced', 0) > 0:
                        logger.info(f"Synced {result['synced']} records")
            except Exception as e:
                logger.error(f"Sync error: {e}")

        logger.info("Sync worker stopped")

    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down...")

        self.is_running = False

        if self.sync_thread:
            self.sync_thread.join(timeout=3)

        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass

        if self.gpio:
            try:
                self.gpio.cleanup()
            except Exception:
                pass

        if self.display:
            try:
                self.display.destroy()
            except Exception:
                pass

        if self.mysql_db:
            try:
                self.mysql_db.close()
            except Exception:
                pass

        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  TrackSite Attendance System")
    print("=" * 70 + "\n")

    try:
        logger.info("Creating system instance...")
        system = AttendanceSystem()

        logger.info("Initializing...")
        if not system.initialize():
            logger.error("Initialization failed!")
            print("\nInitialization failed. Check logs.")
            return 1

        logger.info("System ready!")
        print("\nSystem ready!")
        print("\n" + "=" * 70)
        print("  CONTROLS")
        print("=" * 70)
        print("  - Press 'C' to CONFIRM recognized face")
        print("  - Press 'X' to CANCEL confirmation")
        print("  - Press 'T' to toggle Time-Out mode")
        print("  - Press 'R' to reload encodings")
        print("  - Press 'Q' or ESC to quit")
        print("=" * 70 + "\n")

        system.run()

        print("\nShutdown complete.\n")
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted\n")
        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\nFatal error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())