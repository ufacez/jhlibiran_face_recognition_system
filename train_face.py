"""
Interactive Face Training Script - Auto-retrieves workers from database
Usage: python train_face.py
"""

import cv2
import sys
import numpy as np
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer


def list_workers(mysql_db: MySQLDatabase):
    """Retrieve and display all active workers"""
    print("\n" + "=" * 80)
    print("  AVAILABLE WORKERS")
    print("=" * 80)
    
    workers = mysql_db.fetch_all("""
        SELECT 
            worker_id, 
            worker_code, 
            first_name, 
            last_name, 
            position,
            employment_status
        FROM workers 
        WHERE is_archived = 0
        ORDER BY worker_id ASC
    """)
    
    if not workers:
        print("\n❌ No workers found in database!")
        print("Please add workers through the web dashboard first.")
        return []
    
    print(f"\n{'#':<5} {'ID':<8} {'Code':<12} {'Name':<30} {'Position':<20} {'Status':<10}")
    print("-" * 80)
    
    for idx, worker in enumerate(workers, 1):
        worker_id = worker['worker_id']
        code = worker['worker_code']
        name = f"{worker['first_name']} {worker['last_name']}"
        position = worker['position'] or 'N/A'
        status = worker['employment_status']
        
        print(f"{idx:<5} {worker_id:<8} {code:<12} {name:<30} {position:<20} {status:<10}")
    
    print("=" * 80)
    return workers


def check_existing_encoding(mysql_db: MySQLDatabase, worker_id: int):
    """Check if worker already has face encoding"""
    result = mysql_db.fetch_one("""
        SELECT encoding_id, is_active 
        FROM face_encodings 
        WHERE worker_id = %s
    """, (worker_id,))
    
    return result


def capture_training_images(worker_id: int, worker_name: str, num_images: int = 5):
    """Capture training images from webcam (Optimized)"""
    import face_recognition

    images = []

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n❌ ERROR: Cannot open webcam!")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n📸 Capturing {num_images} training images for: {worker_name}")
    print("=" * 80)
    print("INSTRUCTIONS:")
    print("  ✓ Follow the on-screen positioning guide")
    print("  ✓ Keep your face clearly visible")
    print("  ✓ Ensure good lighting")
    print("  ✓ Press SPACE to capture (only when face detected)")
    print("  ✓ Press Q to quit")
    print("=" * 80)

    # Position guide sequence for better face angles
    position_guides = [
        "Look FORWARD (center)",
        "Look slightly LEFT",
        "Look slightly RIGHT",
        "Look slightly UP",
        "Look slightly DOWN",
        "Look FORWARD again",
        "Slight LEFT + UP",
        "Slight RIGHT + UP",
        "Slight LEFT + DOWN",
        "Look FORWARD (final)"
    ]

    count = 0
    frame_counter = 0
    face_locations = []
    face_landmarks_list = []  # Store landmarks
    DETECT_INTERVAL = 3
    SCALE_FACTOR = 0.25

    while count < num_images:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("\n❌ ERROR: Cannot read from webcam")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_counter += 1
        if frame_counter % DETECT_INTERVAL == 0:
            small_rgb = cv2.resize(rgb_frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            small_locs = face_recognition.face_locations(small_rgb, model="hog")

            face_locations = [
                (int(t / SCALE_FACTOR), int(r / SCALE_FACTOR), 
                 int(b / SCALE_FACTOR), int(l / SCALE_FACTOR))
                for (t, r, b, l) in small_locs
            ]
            
            # Get facial landmarks
            try:
                face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            except Exception:
                face_landmarks_list = []

        face_detected = len(face_locations) > 0

        # Draw face boxes
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Draw facial landmarks mesh CONTINUOUSLY (not just on detect interval)
        if face_detected and face_landmarks_list:
            try:
                for face_landmarks in face_landmarks_list:
                    # Draw facial feature points
                    for feature_name, points in face_landmarks.items():
                        # Draw lines connecting the points
                        for i in range(len(points) - 1):
                            cv2.line(display_frame, points[i], points[i + 1], (0, 255, 255), 1)
                        # Close the loop for certain features
                        if feature_name in ['chin', 'left_eyebrow', 'right_eyebrow', 
                                           'nose_bridge', 'left_eye', 'right_eye',
                                           'top_lip', 'bottom_lip']:
                            if len(points) > 2:
                                cv2.line(display_frame, points[-1], points[0], (0, 255, 255), 1)
                        
                        # Draw small circles on key points
                        for point in points:
                            cv2.circle(display_frame, point, 1, (0, 200, 255), -1)
            except Exception:
                pass

        # Current position guide
        if count < len(position_guides):
            current_guide = position_guides[count]
        else:
            current_guide = "Look FORWARD"
        
        # Status overlay with positioning guide
        if face_detected:
            status_text = "FACE TRACKED - Press SPACE to capture"
            status_color = (0, 255, 0)
        else:
            status_text = "NO FACE - Position yourself"
            status_color = (0, 0, 255)

        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Position guide - LARGE and CENTERED
        cv2.putText(display_frame, current_guide, (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        # Progress bar
        h, w = display_frame.shape[:2]
        progress_bar_width = w - 20
        progress_bar_height = 25
        progress_x = 10
        progress_y = h - 35
        
        # Background of progress bar
        cv2.rectangle(display_frame, (progress_x, progress_y), 
                     (progress_x + progress_bar_width, progress_y + progress_bar_height), 
                     (50, 50, 50), -1)
        
        # Progress fill
        if num_images > 0:
            fill_width = int((count / num_images) * progress_bar_width)
            cv2.rectangle(display_frame, (progress_x, progress_y), 
                         (progress_x + fill_width, progress_y + progress_bar_height), 
                         (0, 255, 0), -1)
        
        # Progress text
        progress_text = f"Progress: {count}/{num_images} ({int(count/num_images*100)}%)"
        cv2.putText(display_frame, progress_text, (progress_x + 5, progress_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(display_frame, f"Worker: {worker_name}", (10, h - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Guide box
        h, w = display_frame.shape[:2]
        guide_w, guide_h = int(w * 0.35), int(h * 0.45)
        left_g = w // 2 - guide_w // 2
        top_g = h // 2 - guide_h // 2
        right_g = left_g + guide_w
        bottom_g = top_g + guide_h
        cv2.rectangle(display_frame, (left_g, top_g), (right_g, bottom_g), (200, 200, 200), 2)

        cv2.imshow("TrackSite - Face Training", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if face_detected:
                images.append(frame.copy())
                count += 1
                print(f"  ✓ Captured image {count}/{num_images} - {current_guide}")

                # Flash effect
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), 30)
                cv2.imshow("TrackSite - Face Training", flash)
                cv2.waitKey(100)
            else:
                print("  ⚠ Cannot capture - No face detected!")

        elif key in (ord("q"), ord("Q")):
            print("\n⚠ Training cancelled by user")
            break

    # COMPLETION SCREEN - Show success message in camera window
    if count >= num_images:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Show countdown from 5 to 1
            for countdown in range(5, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame = cv2.flip(frame, 1)
                
                h, w = frame.shape[:2]
                
                # Create green success overlay
                success_screen = frame.copy()
                overlay = success_screen.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 0), -1)
                cv2.addWeighted(overlay, 0.7, success_screen, 0.3, 0, success_screen)
                
                # Text content
                text1 = "CAPTURE COMPLETE!"
                text2 = f"{count} images captured"
                text3 = f"Training: {worker_name}"
                text4 = "Processing images..."
                text5 = f"Closing in {countdown}..."
                
                # Calculate centered positions
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                (w1, h1), _ = cv2.getTextSize(text1, font, 1.5, 3)
                x1 = (w - w1) // 2
                y1 = h // 2 - 80
                
                (w2, h2), _ = cv2.getTextSize(text2, font, 1.0, 2)
                x2 = (w - w2) // 2
                y2 = h // 2 - 20
                
                (w3, h3), _ = cv2.getTextSize(text3, font, 0.8, 2)
                x3 = (w - w3) // 2
                y3 = h // 2 + 20
                
                (w4, h4), _ = cv2.getTextSize(text4, font, 0.7, 2)
                x4 = (w - w4) // 2
                y4 = h // 2 + 60
                
                (w5, h5), _ = cv2.getTextSize(text5, font, 1.2, 3)
                x5 = (w - w5) // 2
                y5 = h // 2 + 110
                
                # Draw text with shadows
                cv2.putText(success_screen, text1, (x1+3, y1+3), font, 1.5, (0, 0, 0), 5)
                cv2.putText(success_screen, text1, (x1, y1), font, 1.5, (255, 255, 255), 3)
                
                cv2.putText(success_screen, text2, (x2+2, y2+2), font, 1.0, (0, 0, 0), 4)
                cv2.putText(success_screen, text2, (x2, y2), font, 1.0, (255, 255, 255), 2)
                
                cv2.putText(success_screen, text3, (x3+2, y3+2), font, 0.8, (0, 0, 0), 4)
                cv2.putText(success_screen, text3, (x3, y3), font, 0.8, (255, 255, 255), 2)
                
                cv2.putText(success_screen, text4, (x4+2, y4+2), font, 0.7, (0, 0, 0), 3)
                cv2.putText(success_screen, text4, (x4, y4), font, 0.7, (200, 200, 200), 2)
                
                # Countdown - LARGE and BRIGHT YELLOW
                cv2.putText(success_screen, text5, (x5+3, y5+3), font, 1.2, (0, 0, 0), 5)
                cv2.putText(success_screen, text5, (x5, y5), font, 1.2, (0, 255, 255), 3)
                
                # Show frame
                cv2.imshow("TrackSite - Face Training", success_screen)
                cv2.waitKey(1000)  # Wait 1 second per countdown
            
            print("\n✓ All images captured successfully!")

    cap.release()
    cv2.destroyAllWindows()

    return images


def main():
    print("\n" + "=" * 80)
    print("  TrackSite Face Training System")
    print("=" * 80)

    # Initialize database
    mysql_db = MySQLDatabase()
    sqlite_db = SQLiteDatabase()

    print("\n⏳ Connecting to database...")
    if not mysql_db.connect():
        print("\n❌ ERROR: Cannot connect to MySQL database!")
        print("\nTroubleshooting:")
        print("  1. Make sure XAMPP MySQL is running")
        print("  2. Check database 'construction_management' exists")
        print("  3. Verify .env file has correct credentials")
        return 1

    print("✓ Connected to database")

    # List all workers
    workers = list_workers(mysql_db)
    if not workers:
        return 1

    # Get user selection
    print("\n" + "=" * 80)
    while True:
        try:
            selection = input("Enter worker number to train (or 'q' to quit): ").strip()
            
            if selection.lower() == 'q':
                print("\n👋 Exiting...")
                return 0
            
            idx = int(selection)
            
            if idx < 1 or idx > len(workers):
                print(f"❌ Invalid selection! Please enter a number between 1 and {len(workers)}")
                continue
            
            # Get selected worker
            selected_worker = workers[idx - 1]
            break
        
        except ValueError:
            print("❌ Invalid input! Please enter a number or 'q' to quit")
            continue

    worker_id = selected_worker['worker_id']
    worker_name = f"{selected_worker['first_name']} {selected_worker['last_name']}"
    worker_code = selected_worker['worker_code']

    print("\n" + "=" * 80)
    print(f"  SELECTED WORKER")
    print("=" * 80)
    print(f"  Worker:   {worker_id}")
    print(f"  Name:     {worker_name}")
    print(f"  Code:     {worker_code}")
    print(f"  Position: {selected_worker['position']}")
    print("=" * 80)

    # Check if worker already has encoding
    existing = check_existing_encoding(mysql_db, worker_id)
    if existing:
        print(f"\n⚠ WARNING: Worker already has face encoding (ID: {existing['encoding_id']})")
        confirm = input("Do you want to REPLACE it? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("\n👋 Training cancelled")
            return 0
        
        # Deactivate old encoding
        mysql_db.execute_query("""
            UPDATE face_encodings 
            SET is_active = 0 
            WHERE worker_id = %s
        """, (worker_id,))
        print("✓ Old encoding deactivated")

    # Mandatory 10 images for best quality
    num_images = 10
    print("\n" + "=" * 80)
    print(f"  TRAINING: {num_images} images will be captured")
    print("  This ensures the best recognition accuracy!")
    print("=" * 80)

    # Capture images
    images = capture_training_images(worker_id, worker_name, num_images)

    if len(images) < 3:
        print(f"\n❌ ERROR: Need at least 3 images (captured {len(images)})")
        return 1

    print(f"\n⏳ Processing {len(images)} images...")

    # Train face
    recognizer = FaceRecognizer(mysql_db, sqlite_db)
    success = recognizer.train_new_face(images, worker_id)

    if success:
        print("\n" + "=" * 80)
        print("  ✅ FACE TRAINING SUCCESSFUL!")
        print("=" * 80)
        print(f"  Worker: {worker_name}")
        print(f"  Number: {worker_id}")
        print(f"  Code:   {worker_code}")
        print(f"  Images: {len(images)}")
        print("=" * 80)
        print(f"\n✓ {worker_name} can now use facial recognition for attendance")
        
        # Ask if user wants to train another worker
        print("\n" + "=" * 80)
        another = input("Train another worker? (yes/no): ").strip().lower()
        if another in ['yes', 'y']:
            print("\n🔄 Restarting training process...\n")
            main()  # Recursively call main to train another
        else:
            print("\n👋 Exiting training system...")
        
        return 0
    else:
        print("\n❌ Face training failed!")
        print("Please try again with:")
        print("  - Better lighting")
        print("  - Clearer face visibility")
        print("  - Different angles")
        
        # Ask if user wants to retry
        print("\n" + "=" * 80)
        retry = input("Retry training for this worker? (yes/no): ").strip().lower()
        if retry in ['yes', 'y']:
            print("\n🔄 Retrying...\n")
            # Restart from image capture
            images = capture_training_images(worker_id, worker_name, num_images)
            if len(images) >= 3:
                print(f"\n⏳ Processing {len(images)} images...")
                recognizer = FaceRecognizer(mysql_db, sqlite_db)
                success = recognizer.train_new_face(images, worker_id)
                if success:
                    print("\n✅ Training successful on retry!")
                    print(f"✓ {worker_name} can now use facial recognition")
                else:
                    print("\n❌ Training failed again")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())