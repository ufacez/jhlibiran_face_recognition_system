"""
Camera Test Script - Find working camera
Run this first to find which camera and backend works
"""
import cv2

print("\n" + "="*70)
print("  Camera Detection Test")
print("="*70 + "\n")

# Test different backends
backends = [
    (cv2.CAP_ANY, "Default/Auto"),
    (cv2.CAP_MSMF, "Media Foundation (Windows 10+)"),
    (cv2.CAP_DSHOW, "DirectShow (Windows)"),
]

print("Testing camera indices 0-3...\n")

found_camera = False

for index in range(4):
    print(f"\n--- Testing Camera Index {index} ---")
    
    for backend_id, backend_name in backends:
        print(f"\nTrying {backend_name}...")
        
        try:
            cap = cv2.VideoCapture(index, backend_id)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    print(f"  SUCCESS!")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                    print(f"  --> USE: Camera({index}) with {backend_name}")
                    found_camera = True
                    
                    # Show preview
                    print(f"\n  Showing preview (press any key to continue)...")
                    cv2.imshow(f"Camera {index} - {backend_name}", frame)
                    cv2.waitKey(2000)  # Show for 2 seconds
                    cv2.destroyAllWindows()
                else:
                    print(f"  Opened but can't read frames")
                
                cap.release()
            else:
                print(f"  Failed to open")
        
        except Exception as e:
            print(f"  Error: {e}")

print("\n" + "="*70)
if found_camera:
    print("  Found working camera(s) above!")
    print("  Update camera.py if needed to use specific backend")
else:
    print("  No working cameras found!")
    print("  Troubleshooting:")
    print("  1. Close other apps using camera (Zoom, Teams, etc.)")
    print("  2. Check Device Manager for camera issues")
    print("  3. Try external USB camera")
    print("  4. Restart computer")
print("="*70 + "\n")