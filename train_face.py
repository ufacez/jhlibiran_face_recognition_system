"""
Face Training Script - Simplified for PC Development
Usage: python train_face.py --worker_id 1
"""

import argparse
import cv2
import sys
from config.database import MySQLDatabase, SQLiteDatabase
from models.face_recognizer import FaceRecognizer


def capture_training_images(worker_id: int, num_images: int = 5):
    """Capture training images from webcam (Optimized to reduce lag)."""
    import face_recognition

    images = []

    # Try CAP_DSHOW for lower-latency capture on Windows, fall back if not present
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return []

    # Use lower resolution to reduce processing cost (face_recognition works fine)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n📸 Capturing {num_images} training images for worker {worker_id}")
    print("=" * 60)
    print("Instructions:")
    print("  - Look at camera from different angles")
    print("  - Keep your face clearly visible")
    print("  - Ensure good lighting")
    print("  - Press SPACE to capture image (only when face is detected)")
    print("  - Press Q to quit")
    print("=" * 60)

    count = 0
    frame_counter = 0
    face_locations = []
    DETECT_INTERVAL = 3      # detect faces every N frames (tuneable)
    SCALE_FACTOR = 0.25      # detection runs on a scaled-down frame (1/4 size)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from webcam")
            break

         
        
        frame = cv2.flip(frame, 1)
        # Keep a copy for display and saving (do not flip)
        display_frame = frame.copy()

        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_counter += 1
        # Only perform detection every DETECT_INTERVAL frames to reduce CPU usage
        if frame_counter % DETECT_INTERVAL == 0:
            # Resize for faster face detection
            small_rgb = cv2.resize(rgb_frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            small_locs = face_recognition.face_locations(small_rgb, model="hog")

            # Scale locations back to original frame size
            face_locations = [
                (int(t / SCALE_FACTOR), int(r / SCALE_FACTOR), int(b / SCALE_FACTOR), int(l / SCALE_FACTOR))
                for (t, r, b, l) in small_locs
            ]

        face_detected = len(face_locations) > 0

        # Draw detected face boxes (if any)
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Simple, lightweight HUD (fewer draws = faster)
        status_text = "FACE DETECTED - Ready to capture" if face_detected else "NO FACE DETECTED - Position yourself"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)

        cv2.putText(display_frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(display_frame, f"Captured: {count}/{num_images}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(display_frame, "SPACE = Capture  |  Q = Quit",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw a faint center guide rectangle to help alignment (cheap draw)
        h, w = display_frame.shape[:2]
        guide_w, guide_h = int(w * 0.35), int(h * 0.45)
        left_g = w // 2 - guide_w // 2
        top_g = h // 2 - guide_h // 2
        right_g = left_g + guide_w
        bottom_g = top_g + guide_h
        guide_color = (200, 200, 200)
        cv2.rectangle(display_frame, (left_g, top_g), (right_g, bottom_g), guide_color, 1)
        cv2.putText(display_frame, "Align face inside guide", (left_g, top_g - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, guide_color, 1)

        cv2.imshow("TrackSite - Face Training (Optimized)", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # SPACE to capture
            if face_detected:
                # Save the original non-flipped high-res capture (we used a low-res preview)
                images.append(frame.copy())
                count += 1
                print(f"✓ Captured image {count}/{num_images}")

                # Short visual feedback flash
                flash = display_frame.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), 30)
                cv2.imshow("TrackSite - Face Training (Optimized)", flash)
                cv2.waitKey(80)
            else:
                print("⚠ Cannot capture — no face detected. Please align your face inside the guide.")

        elif key in (ord("q"), ord("Q")):
            print("\n⚠ Training cancelled by user")
            break

    cap.release()
    cv2.destroyAllWindows()

    return images


def main():
    parser = argparse.ArgumentParser(description='Train facial recognition for a worker')
    parser.add_argument('--worker_id', type=int, required=True,
                        help='Worker ID from database')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of training images to capture (default: 5)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TrackSite Face Training System")
    print("=" * 60)

    # Initialize databases
    mysql_db = MySQLDatabase()
    sqlite_db = SQLiteDatabase()

    if not mysql_db.connect():
        print("\n❌ ERROR: Cannot connect to MySQL database!")
        print("Make sure:")
        print("  1. XAMPP MySQL is running")
        print("  2. Database 'construction_management' exists")
        print("  3. .env file has correct credentials")
        return 1

    # Verify worker exists
    worker = mysql_db.fetch_one(
        "SELECT * FROM workers WHERE worker_id = %s",
        (args.worker_id,)
    )

    if not worker:
        print(f"\n❌ ERROR: Worker ID {args.worker_id} not found in database!")
        print("Please add the worker through the web dashboard first.")
        return 1

    print(f"\n✓ Found worker: {worker['first_name']} {worker['last_name']}")
    print(f"  Worker Code: {worker['worker_code']}")
    print(f"  Position: {worker['position']}")

    # Capture images
    images = capture_training_images(args.worker_id, args.num_images)

    if len(images) < 3:
        print(f"\n❌ ERROR: Need at least 3 images (captured {len(images)})")
        return 1

    print(f"\n⏳ Processing {len(images)} images...")

    # Train face
    recognizer = FaceRecognizer(mysql_db, sqlite_db)
    success = recognizer.train_new_face(images, args.worker_id)

    if success:
        print("\n" + "=" * 60)
        print("✅ Face training successful!")
        print("=" * 60)
        print(f"Worker {worker['first_name']} {worker['last_name']} can now use facial recognition")
        return 0
    else:
        print("\n❌ Face training failed!")
        print("Please try again with clearer images")
        return 1


if __name__ == "__main__":
    sys.exit(main())
