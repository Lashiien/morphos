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
        "models/trained/best.pt",  # Primary location after refactoring
        "_archive_old_data/runs/detect/new_training/morphos_v24/weights/best.pt",  # Best trained weights
        "_archive_old_data/runs/detect/new_training/morphos_v2/weights/best.pt",  # Alternative trained weights
        "_archive_old_data/runs/detect/morphos_training/exp_fixed/weights/best.pt",  # Original trained weights
        "best.pt"  # Last resort fallback
    ]
    CONFIDENCE_THRESHOLD = 0.5
    CONF_THRESHOLD = 0.5  # alias for compatibility
    
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
    # Buffer size (frames to analyze)
    FLASH_BUFFER_SIZE = 30  # 1 second @ 30fps
    
    # Variance threshold for color channels
    FLASH_THRESHOLD_STD = 15.0  # Balanced sensitivity
    
    # Emergency light frequency range (Hz)
    FLASH_FREQUENCY_MIN = 1.0
    FLASH_FREQUENCY_MAX = 6.0  # Extended to catch 3Hz + 6Hz harmonics
    FLASH_FREQUENCY_RANGE = (1.0, 6.0)  # tuple for direct use
    
    # Confirmation requirements
    FLASH_CONFIRMATION_FRAMES = 25  # ~0.8 seconds of continuous flashing
    FLASH_CONFIRMATION_THRESHOLD_FRAMES = FLASH_CONFIRMATION_FRAMES  # alias for paper-ready naming
    FRAMES_TO_CLEAR_EMERGENCY = 30  # 1 second without detection to clear
    FRAMES_TO_CLEAR = 30  # alias for compatibility
    FLASH_LOST_FRAMES = 60  # 2 seconds without flash to clear (car visible but no flash)
    
    # ========================================================================
    # SERIAL COMMUNICATION
    # ========================================================================
    SERIAL_BAUD_RATE = 9600
    SERIAL_TIMEOUT = 1
    SERIAL_RETRY_ATTEMPTS = 3
    SERIAL_RETRY_DELAY = 2  # seconds
    # Aliases for compatibility with morphos_final_v2.py
    BAUD_RATE = SERIAL_BAUD_RATE
    SERIAL_RETRY_ATTEMPTS = SERIAL_RETRY_ATTEMPTS
    SERIAL_RETRY_DELAY = SERIAL_RETRY_DELAY
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_TO_FILE = True
    LOG_FILE = "morphos_system.log"
    
    @classmethod
    def get_model_path(cls):
        """Find first existing model"""
        for path in cls.MODEL_PATHS:
            if Path(path).exists():
                return path
        raise FileNotFoundError(f"No model found in: {cls.MODEL_PATHS}")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        assert cls.CAMERA_FPS > 0, "FPS must be positive"
        assert cls.FLASH_BUFFER_SIZE > 0, "Buffer size must be positive"
        assert cls.FLASH_FREQUENCY_MIN < cls.FLASH_FREQUENCY_MAX, "Invalid frequency range"
        return True