import cv2
import logging
from threading import Thread, Lock
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Optimized camera with threaded reading for 30 FPS"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.is_running = False
        
        # Threading
        self.lock = Lock()
        self.thread: Optional[Thread] = None
    
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            logger.info(f"Opening camera {self.camera_index}...")
            
            # Use CAP_ANY (works best based on your test)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            logger.info("Camera opened successfully")
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # Test read
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read test frame")
                return False
            
            # Start threaded reading
            self.is_running = True
            self.thread = Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            
            logger.info("Camera initialized")
            logger.info(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            logger.info(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
            
            return True
        
        except Exception as e:
            logger.exception(f"Camera error: {e}")
            return False
    
    def _read_frames(self):
        """Background thread for reading frames"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                with self.lock:
                    self.ret = ret
                    self.frame = frame
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame (non-blocking)"""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return False, None
    
    def set_resolution(self, width: int, height: int):
        """Set resolution"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Resolution set to {width}x{height}")
    
    def set_fps(self, fps: int):
        """Set FPS"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            logger.info(f"FPS set to {fps}")
    
    def release(self):
        """Release camera"""
        logger.info("Releasing camera...")
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Camera released")