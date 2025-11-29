"""Test configuration settings"""
from config.settings import Config

print("Testing Configuration Settings")
print("=" * 519)

# Test MySQL settings
print("\n📊 MySQL Configuration:")
print(f"   Host: {Config.MYSQL_HOST} (type: {type(Config.MYSQL_HOST).__name__})")
print(f"   Port: {Config.MYSQL_PORT} (type: {type(Config.MYSQL_PORT).__name__})")
print(f"   User: {Config.MYSQL_USER}")
print(f"   Database: {Config.MYSQL_DATABASE}")

# Test Face Recognition settings
print("\n🤖 Face Recognition Settings:")
print(f"   Tolerance: {Config.FACE_RECOGNITION_TOLERANCE}")
print(f"   Detection Model: {Config.FACE_DETECTION_MODEL}")
print(f"   Min Face Size: {Config.MIN_FACE_SIZE}")

# Test Hardware settings
print("\n🎥 Hardware Settings:")
print(f"   Camera Resolution: {Config.CAMERA_RESOLUTION}")
print(f"   Camera Framerate: {Config.CAMERA_FRAMERATE}")
print(f"   GPIO Button Pin: {Config.GPIO_TIMEOUT_BUTTON}")

# Test Logging
print("\n📝 Logging Configuration:")
print(f"   Log Level: {Config.LOG_LEVEL}")
print(f"   Log File: {Config.LOG_FILE}")

print("\n" + "=" * 60)
print("✅ All configuration settings loaded successfully!")

# Verify types
print("\n🔍 Type Validation:")
assert isinstance(Config.MYSQL_HOST, str), "MYSQL_HOST should be string"
assert isinstance(Config.MYSQL_PORT, int), "MYSQL_PORT should be integer"
assert isinstance(Config.FACE_RECOGNITION_TOLERANCE, float), "Tolerance should be float"
assert isinstance(Config.CAMERA_RESOLUTION, tuple), "Resolution should be tuple"
print("   ✅ All types are correct!")