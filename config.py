"""
PROJECT MORPHOS - CONFIGURATION FILE
Centralized settings for easy tuning
"""
from pathlib import Path


class MorphosConfig:
    # ========================================================================
    # MODEL SETTINGS
    # ========================================================================
    MODEL_PATHS = [
        "models/trained/best.pt",
        "best.pt",
    ]
    CONFIDENCE_THRESHOLD = 0.5

    # ========================================================================
    # CAMERA SETTINGS
    # ========================================================================
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30.0

    # ========================================================================
    # FLASH DETECTION PARAMETERS
    # ========================================================================
    FLASH_BUFFER_SIZE = 30              # 1 second @ 30fps
    FLASH_THRESHOLD_STD = 15.0          # Variance threshold for color channels
    FLASH_FREQUENCY_RANGE = (1.0, 6.0)  # Emergency light Hz range (includes harmonics)
    FLASH_CONFIRMATION_FRAMES = 25      # ~0.8 seconds of continuous flashing to confirm
    FRAMES_TO_CLEAR = 30                # Frames without detection to exit emergency
    FLASH_LOST_FRAMES = 60              # Frames without flash to exit emergency

    # ========================================================================
    # SERIAL COMMUNICATION
    # ========================================================================
    SERIAL_BAUD_RATE = 9600
    SERIAL_TIMEOUT = 1
    SERIAL_RETRY_ATTEMPTS = 3
    SERIAL_RETRY_DELAY = 2              # seconds

    # ========================================================================
    # ARDUINO WATCHDOG KEEPALIVE
    # ========================================================================
    HEARTBEAT_INTERVAL = 2.0            # Seconds between b'1' keepalives during emergency

    @classmethod
    def get_model_path(cls):
        for path in cls.MODEL_PATHS:
            if Path(path).exists():
                return path
        raise FileNotFoundError(f"No model found in: {cls.MODEL_PATHS}")

    @classmethod
    def validate(cls):
        assert cls.CAMERA_FPS > 0, "FPS must be positive"
        assert cls.FLASH_BUFFER_SIZE > 0, "Buffer size must be positive"
        assert cls.FLASH_FREQUENCY_RANGE[0] < cls.FLASH_FREQUENCY_RANGE[1], "Invalid frequency range"
        return True
