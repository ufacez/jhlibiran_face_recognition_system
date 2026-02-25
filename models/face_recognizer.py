import face_recognition
import numpy as np
import cv2
import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from config.settings import Config
from config.database import MySQLDatabase, SQLiteDatabase

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Optimized face recognition - smooth 30 FPS"""
    
    def __init__(self, mysql_db: MySQLDatabase, sqlite_db: SQLiteDatabase):
        self.mysql_db = mysql_db
        self.sqlite_db = sqlite_db
        self.known_encodings: List[np.ndarray] = []
        self.known_metadata: List[Dict[str, Any]] = []
        self.last_update: Optional[float] = None
        
        # Performance settings - OPTIMIZED for smooth tracking
        self.scale_factor = 0.35  # Further reduced for faster processing
        self.tolerance = 0.5
        
        # Cache last face locations to maintain smooth tracking
        self.last_face_locations = []
        self.last_face_names = []
        self.last_face_ids = []  # Track worker IDs
    
    def load_encodings(self) -> int:
        """Load face encodings from database"""
        logger.info("Loading face encodings...")
        
        # Try MySQL first
        encodings = []
        if self.mysql_db and self.mysql_db.is_connected:
            encodings = self._load_from_mysql()
            if encodings and self.sqlite_db:
                self.sqlite_db.cache_face_encodings(encodings)
        else:
            # Fallback to SQLite
            if self.sqlite_db:
                encodings = self.sqlite_db.get_cached_encodings()
                logger.warning("Using cached encodings (offline)")
        
        # Parse encodings
        self.known_encodings = []
        self.known_metadata = []
        
        for enc_data in encodings:
            try:
                encoding_array = np.array(json.loads(enc_data['encoding_data']))
                self.known_encodings.append(encoding_array)
                
                self.known_metadata.append({
                    'worker_id': enc_data['worker_id'],
                    'first_name': enc_data['first_name'],
                    'last_name': enc_data['last_name'],
                    'worker_code': enc_data['worker_code']
                })
            except Exception as e:
                logger.error(f"Failed to parse encoding: {e}")
        
        logger.info(f"Loaded {len(self.known_encodings)} encodings")
        return len(self.known_encodings)
    
    def _load_from_mysql(self) -> List[Dict[str, Any]]:
        """Load from MySQL"""
        query = """
            SELECT 
                fe.encoding_id,
                fe.worker_id,
                fe.encoding_data,
                w.first_name,
                w.last_name,
                w.worker_code,
                fe.is_active
            FROM face_encodings fe
            JOIN workers w ON fe.worker_id = w.worker_id
            WHERE fe.is_active = 1 
            AND w.employment_status = 'active'
            AND w.is_archived = 0
        """
        return self.mysql_db.fetch_all(query) if self.mysql_db else []
    
    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        SMOOTH CONTINUOUS face recognition - tracks faces every frame
        
        Returns:
            (worker_info, annotated_frame, face_box) or (None, frame_with_all_faces, None)
        """
        if not self.known_encodings:
            return None, frame, None
        
        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect ALL faces - FAST mode
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog',
            number_of_times_to_upsample=0
        )
        
        # If no faces found, keep showing last known faces briefly
        if not face_locations:
            # Draw last known faces (faded)
            for i, (top, right, bottom, left) in enumerate(self.last_face_locations):
                if i < len(self.last_face_names):
                    name = self.last_face_names[i]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 150, 0), 2)
                    cv2.putText(frame, name, (left, max(25, top - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
            return None, frame, None
        
        # Get encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Get facial landmarks for mesh visualization
        face_landmarks_list = []
        try:
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        except Exception:
            pass  # Skip on error
        
        # Scale back to original size
        scale_reciprocal = 1.0 / self.scale_factor
        
        first_recognized_worker = None
        first_face_box = None
        
        current_face_locations = []
        current_face_names = []
        current_face_ids = []
        
        # Process each face
        for face_idx, ((top, right, bottom, left), encoding) in enumerate(zip(face_locations, face_encodings)):
            # Scale coordinates
            top = int(top * scale_reciprocal)
            right = int(right * scale_reciprocal)
            bottom = int(bottom * scale_reciprocal)
            left = int(left * scale_reciprocal)
            
            current_face_locations.append((top, right, bottom, left))
            
            # Draw facial landmarks mesh for this face (if available)
            if face_idx < len(face_landmarks_list):
                landmarks = face_landmarks_list[face_idx]
                for feature_name, points in landmarks.items():
                    # Scale landmark points to full resolution
                    scaled_points = [(int(x * scale_reciprocal), int(y * scale_reciprocal)) for x, y in points]
                    
                    # Draw lines connecting the points
                    for i in range(len(scaled_points) - 1):
                        cv2.line(frame, scaled_points[i], scaled_points[i + 1], (255, 200, 0), 1)
                    
                    # Close the loop for certain features
                    if feature_name in ['chin', 'left_eyebrow', 'right_eyebrow', 
                                       'nose_bridge', 'left_eye', 'right_eye',
                                       'top_lip', 'bottom_lip']:
                        if len(scaled_points) > 2:
                            cv2.line(frame, scaled_points[-1], scaled_points[0], (255, 200, 0), 1)
                    
                    # Draw small dots on key points
                    for point in scaled_points:
                        cv2.circle(frame, point, 1, (255, 220, 50), -1)
            
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings,
                encoding,
                tolerance=self.tolerance
            )
            
            if True not in matches:
                # Unknown - draw red box CONTINUOUSLY
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
                
                # Shadow
                cv2.putText(frame, "Unknown", (left + 2, max(25, top - 10) + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                # Main text
                cv2.putText(frame, "Unknown", (left, max(25, top - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                current_face_names.append("Unknown")
                current_face_ids.append(None)
                continue
            
            # Best match
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                encoding
            )
            best_match_idx = np.argmin(face_distances)
            
            if matches[best_match_idx]:
                worker_info = self.known_metadata[best_match_idx].copy()
                confidence = 1 - face_distances[best_match_idx]
                worker_info['confidence'] = confidence
                
                # Draw GREEN box CONTINUOUSLY - THICK and BRIGHT
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 5)
                
                # Draw NAME above box
                first = worker_info.get("first_name") or ""
                last = worker_info.get("last_name") or ""
                name = f"{first} {last}".strip() or "Unknown"
                
                current_face_names.append(name)
                current_face_ids.append(worker_info.get('worker_id'))
                
                label_y = max(30, top - 10)
                
                # Shadow for readability
                cv2.putText(frame, name, (left + 2, label_y + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                # Main text - BRIGHT GREEN
                cv2.putText(frame, name, (left, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Store first recognized worker for confirmation system
                if first_recognized_worker is None:
                    first_recognized_worker = worker_info
                    first_face_box = (top, right, bottom, left)
        
        # Update cache - ALWAYS maintain tracking
        self.last_face_locations = current_face_locations
        self.last_face_names = current_face_names
        self.last_face_ids = current_face_ids
        
        # Return first recognized worker (if any) for confirmation
        return first_recognized_worker, frame, first_face_box
    
    def train_new_face(self, images: List[np.ndarray], worker_id: int) -> bool:
        """Train new face"""
        encodings = []
        
        logger.info(f"Training face for worker {worker_id}...")
        
        for idx, img in enumerate(images):
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                logger.warning(f"No face in image {idx+1}")
                continue
            
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces in image {idx+1}")
            
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if face_encodings:
                encodings.append(face_encodings[0])
                logger.info(f"✓ Processed image {idx+1}")
        
        if len(encodings) < 3:
            logger.error(f"Need 3+ images (got {len(encodings)})")
            return False
        
        # Average encodings
        avg_encoding = np.mean(encodings, axis=0)
        encoding_json = json.dumps(avg_encoding.tolist())
        
        # Store
        if not self.mysql_db or not self.mysql_db.is_connected:
            logger.error("MySQL not connected")
            return False
        
        query = """
            INSERT INTO face_encodings 
            (worker_id, encoding_data, is_active)
            VALUES (%s, %s, 1)
        """
        encoding_id = self.mysql_db.execute_query(query, (worker_id, encoding_json))
        
        if encoding_id:
            logger.info(f"✅ Trained worker {worker_id}")
            self.load_encodings()
            return True
        else:
            logger.error("Failed to store encoding")
            return False