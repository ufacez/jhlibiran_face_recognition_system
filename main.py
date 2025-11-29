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
    """Main attendance system - 30 FPS optimized"""
    
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
        self.skip_frames = 2
        self.frame_counter = 0
        
        # Threading
        self.sync_thread: Optional[threading.Thread] = None
        
        # Success display overlay (non-blocking)
        self.success_overlay: Optional[Dict[str, Any]] = None
        self.overlay_lock = threading.Lock()
        self.overlay_end_time: Optional[float] = None

        # Confirmation system
        self.pending_worker = None
        self.waiting_for_confirmation = False

    
    def initialize(self) -> bool:
        """Initialize all system components"""
        logger.info("Initializing components...")
        
        try:
            # Initialize databases
            logger.info("Connecting to databases...")
            self.mysql_db = MySQLDatabase()
            self.sqlite_db = SQLiteDatabase()
            
            mysql_connected = self.mysql_db.connect()
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
            self.gpio.add_button_callback(self._handle_timeout_button)
            logger.info("GPIO initialized")
            
            # Create display window (WINDOWED)
            logger.info("Initializing display...")
            self.display = Display()
            self.display.create_window(fullscreen=False)
            logger.info("Display initialized (windowed)")
            
            # Load face encodings
            logger.info("Loading face encodings...")
            encoding_count = self.face_recognizer.load_encodings()
            if encoding_count == 0:
                logger.warning("No face encodings loaded")
            else:
                logger.info(f"Loaded {encoding_count} face encodings")
            
            logger.info("="*60)
            logger.info("System ready!")
            logger.info("="*60)
            return True
            
        except Exception as e:
            logger.exception(f"Initialization error: {e}")
            return False
    
    def run(self):
        """Main loop - optimized for 30 FPS"""
        logger.info("Starting attendance system...")
        self.is_running = True
        
        # Start background sync
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        try:
            last_frame_time = time.time()
            fps_values = []
            current_fps = 0
            
            while self.is_running:
                loop_start = time.time()
                
                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret or frame is None:
                    logger.error("Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.flip(frame, 1)
                self.frame_counter += 1
                
                # Process recognition (every N frames for performance)
                worker_info = None
                if self.frame_counter % self.skip_frames == 0 and self.success_overlay is None:
                    worker_info, frame = self.face_recognizer.recognize_face(frame)
                    
                    if worker_info:
                        self._handle_recognition(worker_info)
                
                worker_info, frame, face_box = self.face_recognizer.recognize_face(frame)
if worker_info:
    worker_info['face_box'] = face_box
    self._handle_recognition(worker_info)

                # Add status overlay
                status = self._get_status_text(current_fps)
                frame = self.display.add_status_bar(frame, status)
                
                # Add mode indicator
                if self.timeout_mode:
                    frame = self.display.add_overlay(
                        frame,
                        "TIME-OUT MODE",
                        position=(50, 50),
                        color=(0, 165, 255),
                        font_scale=1.5
                    )
                
                # Draw success overlay if active (NON-BLOCKING)
                with self.overlay_lock:
                    if self.success_overlay is not None:
                        if time.time() < self.overlay_end_time:
                            frame = self._draw_success_overlay(frame, self.success_overlay)
                        else:
                            self.success_overlay = None
                            self.overlay_end_time = None
                
                # Display frame
                self.display.show_frame(frame)
                

                # If waiting for confirmation — draw green box & name
                if self.waiting_for_confirmation and self.pending_worker:
                    worker = self.pending_worker
                    box = worker.get('face_box')  # you must add this inside recognizer
                    if box:
                        (top, right, bottom, left) = box
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                        cv2.putText(frame, f"{worker['first_name']} {worker['last_name']} (Press C)",
                                    (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        
                # FPS calculation
                fps_values.append(1.0 / (time.time() - last_frame_time))
                last_frame_time = time.time()
                
                if len(fps_values) >= 10:
                    current_fps = int(sum(fps_values) / len(fps_values))
                    fps_values = []
                
                # Handle keyboard
                key = self.display.wait_key(1)
                if key == ord('q') or key == 27:
                    logger.info("Quit key pressed")
                    break
                elif key == ord('t'):
                    self._toggle_timeout_mode()
                elif key == ord('r'):
                    self._reload_encodings()
                # Confirm recognition
                elif key == ord('c') and self.waiting_for_confirmation and self.pending_worker:
                    logger.info("User confirmed recognition")
                    result = self._process_attendance(self.pending_worker)
                    worker_name = f"{self.pending_worker['first_name']} {self.pending_worker['last_name']}"
                    self._show_result_overlay(result, worker_name)

                    # Reset confirmation state
                    self.pending_worker = None
                    self.waiting_for_confirmation = False

                
                # Frame limiting
                elapsed = time.time() - loop_start
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
        finally:
            self.shutdown()
    
    def _handle_recognition(self, worker_info: Dict[str, Any]):
        """Store recognized worker — DO NOT auto log attendance"""
        now = datetime.now()

        # Cooldown
        if self.last_recognition_time:
            if (now - self.last_recognition_time).total_seconds() < 3:
                return

        self.last_recognition_time = now
        self.pending_worker = worker_info
        self.waiting_for_confirmation = True

        worker_name = f"{worker_info['first_name']} {worker_info['last_name']}"
        logger.info(f"Recognized (waiting for confirmation): {worker_name}")
        
        result = self._process_attendance(worker_info)
        self._show_result_overlay(result, worker_name)
    
    def _process_attendance(self, worker_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process attendance (synchronous, fast)"""
        worker_id = worker_info['worker_id']
        
        if self.timeout_mode:
            result = self.attendance_logger.log_timeout(worker_id)
            if result['success']:
                self.timeout_mode = False
                self.gpio.set_led(False)
        else:
            result = self.attendance_logger.log_timein(worker_id)
        
        return result
    
    def _show_result_overlay(self, result: Dict[str, Any], worker_name: str):
        """Set success overlay data (non-blocking)"""
        current_time = datetime.now()
        
        overlay_data = {
            'worker_name': worker_name,
            'result': result,
            'timestamp': current_time
        }
        
        with self.overlay_lock:
            self.success_overlay = overlay_data
            self.overlay_end_time = time.time() + 3.0
    
    def _draw_success_overlay(self, frame: np.ndarray, overlay_data: Dict[str, Any]) -> np.ndarray:
        """Draw success overlay on frame"""
        result = overlay_data['result']
        worker_name = overlay_data['worker_name']
        timestamp = overlay_data['timestamp']
        
        h, w = frame.shape[:2]
        overlay_frame = frame.copy()
        
        # Dark background
        overlay_bg = overlay_frame.copy()
        cv2.rectangle(overlay_bg, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.7, overlay_frame, 0.3, 0, overlay_frame)
        
        if result['success']:
            action = result.get('action', 'unknown')
            
            if action == 'timein':
                status_icon = "TIME IN"
                time_str = result.get('time_in', timestamp.strftime('%I:%M:%S %p'))
                detail_text = f"Clocked In: {time_str}"
                icon_color = (0, 255, 0)
                bg_color = (0, 120, 0)
            elif action == 'timeout':
                status_icon = "TIME OUT"
                hours = result.get('hours_worked', '0.00')
                time_str = timestamp.strftime('%I:%M:%S %p')
                detail_text = f"Time: {time_str} | Hours: {hours}"
                icon_color = (0, 200, 255)
                bg_color = (0, 80, 120)
            elif action == 'already_in':
                status_icon = "ALREADY IN"
                detail_text = "See supervisor"
                icon_color = (0, 255, 255)
                bg_color = (0, 120, 120)
            elif action == 'completed':
                status_icon = "DONE"
                detail_text = "Thank you!"
                icon_color = (255, 200, 0)
                bg_color = (120, 100, 0)
            else:
                status_icon = "SUCCESS"
                detail_text = timestamp.strftime('%I:%M:%S %p')
                icon_color = (0, 255, 0)
                bg_color = (0, 120, 0)
            
            # Top banner
            cv2.rectangle(overlay_frame, (0, 0), (w, 60), bg_color, -1)
            cv2.rectangle(overlay_frame, (0, 0), (w, 60), icon_color, 3)
            
            # Status
            text_size = cv2.getTextSize(status_icon, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, status_icon, (text_x, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Worker name
            text_size = cv2.getTextSize(worker_name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, worker_name, (text_x, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Detail
            text_size = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, detail_text, (text_x, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, icon_color, 2)
            
            # Checkmark
            center_x = w // 2
            if action in ['timein', 'timeout']:
                cv2.circle(overlay_frame, (center_x, 230), 50, icon_color, 5)
                pts = np.array([
                    [center_x - 20, 230],
                    [center_x - 5, 245],
                    [center_x + 25, 215]
                ], np.int32)
                cv2.polylines(overlay_frame, [pts], False, icon_color, 8)
            
            # Date
            date_str = timestamp.strftime('%b %d, %Y')
            text_size = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, date_str, (text_x, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            
            # Time
            time_display = timestamp.strftime('%I:%M:%S %p')
            text_size = cv2.getTextSize(time_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, time_display, (text_x, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        else:
            # Error overlay
            cv2.rectangle(overlay_frame, (0, 0), (w, 60), (0, 80, 120), -1)
            cv2.rectangle(overlay_frame, (0, 0), (w, 60), (0, 165, 255), 3)
            
            text_size = cv2.getTextSize("PLEASE TRY AGAIN", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, "PLEASE TRY AGAIN", (text_x, 42),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            message = result.get('message', 'Unknown error')
            friendly_messages = {
                'Already scanned recently': 'Already recorded.\nPlease wait.',
                'Attendance already completed today': 'All done for today.\nThank you!',
                'No time-in found for today': 'Please clock in first.',
                'Already timed in. Ready for time-out?': 'Already clocked in.\nSee supervisor.'
            }
            
            display_message = friendly_messages.get(message, message)
            lines = display_message.split('\n')
            y_pos = 120
            for line in lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(overlay_frame, line, (text_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_pos += 40
            
            # X mark
            center_x = w // 2
            center_y = 230
            cv2.circle(overlay_frame, (center_x, center_y), 50, (0, 165, 255), 5)
            cv2.line(overlay_frame, (center_x - 20, center_y - 20), 
                    (center_x + 20, center_y + 20), (0, 165, 255), 8)
            cv2.line(overlay_frame, (center_x + 20, center_y - 20), 
                    (center_x - 20, center_y + 20), (0, 165, 255), 8)
            
            # Time
            time_str = timestamp.strftime('%I:%M:%S %p')
            text_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(overlay_frame, time_str, (text_x, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return overlay_frame
    
    def _toggle_timeout_mode(self):
        """Toggle time-out mode"""
        self.timeout_mode = not self.timeout_mode
        self.gpio.set_led(self.timeout_mode)
        mode_text = "TIME-OUT MODE" if self.timeout_mode else "TIME-IN MODE"
        logger.info(mode_text)
    
    def _handle_timeout_button(self):
        """GPIO button callback"""
        logger.info("Timeout button pressed")
        self._toggle_timeout_mode()
    
    def _reload_encodings(self):
        """Reload face encodings"""
        logger.info("Reloading face encodings...")
        count = self.face_recognizer.load_encodings()
        logger.info(f"Reloaded {count} faces")
    
    def _get_status_text(self, fps: int = 0) -> str:
        """Get status text"""
        parts = []
        
        if self.mysql_db and self.mysql_db.is_connected:
            parts.append("[ONLINE]")
        else:
            parts.append("[OFFLINE]")
        
        if self.timeout_mode:
            parts.append("TIME-OUT")
        else:
            parts.append("TIME-IN")
        
        parts.append(f"{fps} FPS")
        
        now = datetime.now()
        parts.append(now.strftime('%I:%M:%S %p'))
        parts.append(now.strftime('%b %d, %Y'))
        
        return " | ".join(parts)
    
    def _sync_worker(self):
        """Background sync worker"""
        logger.info("Sync worker started")
        
        while self.is_running:
            time.sleep(Config.SYNC_INTERVAL_SECONDS)
            
            try:
                if self.mysql_db and not self.mysql_db.is_connected:
                    if self.mysql_db.connect():
                        logger.info("MySQL reconnected")
                        self.face_recognizer.load_encodings()
                
                if self.sync_manager:
                    result = self.sync_manager.sync_all()
                    if result['synced'] > 0:
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
            self.camera.release()
        
        if self.gpio:
            self.gpio.cleanup()
        
        if self.display:
            self.display.destroy()
        
        if self.mysql_db:
            self.mysql_db.close()
        
        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  TrackSite Attendance System")
    print("="*70 + "\n")
    
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
        print("\n" + "="*70)
        print("  CONTROLS")
        print("="*70)
        print("  - Press 'q' or ESC to quit")
        print("  - Press 't' to toggle Time-Out mode")
        print("  - Press 'r' to reload encodings")
        print("="*70 + "\n")
        
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