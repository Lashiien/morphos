import cv2
from ultralytics import YOLO
import serial
import serial.tools.list_ports
import time
import sys
import numpy as np
from collections import deque
import logging
from pathlib import Path

# Import our advanced flash detector
from flash_detector import EmergencyFlashDetector
from config import MorphosConfig as Config

# LOGGING SETUP
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (now imported from config.py)
# ============================================================================

# ============================================================================
# SERIAL COMMUNICATION WITH ERROR HANDLING
# ============================================================================
class ArduinoController:
    """Robust Arduino serial communication with auto-reconnect"""
    
    def __init__(self, baud_rate=9600, timeout=1):
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_port = None
        self.is_connected = False
        self.current_state = False  # False = Normal, True = Emergency
        
    def connect(self, port=None):
        """Connect to Arduino with auto-detection"""
        logger.info("Scanning serial ports...")
        ports = list(serial.tools.list_ports.comports())
        
        if not ports:
            logger.warning("No serial ports found. Running in DEBUG mode.")
            return None
        
        logger.info("Available ports:")
        for i, p in enumerate(ports):
            logger.info(f"  {i+1}. {p.device} - {p.description}")
        
        # Auto-detect Arduino
        arduino_port = port
        if not arduino_port:
            for p in ports:
                if any(keyword in p.description for keyword in 
                      ["Arduino", "CH340", "USB-SERIAL", "COM"]):
                    arduino_port = p.device
                    logger.info(f"Auto-detected: {arduino_port}")
                    break
        
        if not arduino_port and len(ports) == 1:
            arduino_port = ports[0].device
            logger.info(f"Using only available port: {arduino_port}")
        
        if not arduino_port:
            choice = input("\nEnter port (COM3/COM4/etc) or number: ").strip()
            if choice.isdigit() and 0 < int(choice) <= len(ports):
                arduino_port = ports[int(choice)-1].device
            else:
                arduino_port = choice
        
        # Attempt connection with retry
        for attempt in range(Config.SERIAL_RETRY_ATTEMPTS):
            try:
                self.serial_port = serial.Serial(
                    arduino_port, 
                    self.baud_rate, 
                    timeout=self.timeout
                )
                time.sleep(2)  # Wait for Arduino reset
                self.is_connected = True
                logger.info(f"✓ Connected to Arduino on {arduino_port}")
                return self.serial_port
            
            except serial.SerialException as e:
                logger.error(f"Connection attempt {attempt+1} failed: {e}")
                if attempt < Config.SERIAL_RETRY_ATTEMPTS - 1:
                    logger.info(f"Retrying in {Config.SERIAL_RETRY_DELAY}s...")
                    time.sleep(Config.SERIAL_RETRY_DELAY)
        
        logger.warning("Failed to connect. Running in DEBUG mode.")
        return None
    
    def send_command(self, command: bytes) -> bool:
        """Send command with error handling and auto-reconnect"""
        if not self.serial_port:
            logger.debug(f"DEBUG MODE: Would send {command}")
            return True
        
        try:
            self.serial_port.write(command)
            self.serial_port.flush()
            return True
        
        except serial.SerialException as e:
            logger.error(f"Serial connection lost: {e}")
            self.is_connected = False
            
            # Attempt reconnection
            logger.info("Attempting to reconnect...")
            if self.reconnect():
                try:
                    self.serial_port.write(command)
                    self.serial_port.flush()
                    return True
                except:
                    pass
            return False
        
        except Exception as e:
            logger.error(f"Unexpected serial error: {e}")
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to Arduino"""
        try:
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
            self.serial_port = self.connect()
            return self.is_connected
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
    
    def activate_emergency(self) -> bool:
        """Activate emergency mode (Green light)"""
        if not self.current_state:
            if self.send_command(b'1'):
                self.current_state = True
                logger.info("🚨 EMERGENCY ACTIVATED → Traffic going GREEN")
                return True
        return self.current_state
    
    def clear_emergency(self) -> bool:
        """Return to normal traffic cycle"""
        if self.current_state:
            if self.send_command(b'0'):
                self.current_state = False
                logger.info("✓ EMERGENCY CLEARED → Normal cycle resumed")
                return False
        return self.current_state
    
    def force_normal_mode(self):
        """Force normal mode (cleanup on exit)"""
        if self.current_state:
            self.send_command(b'0')
            self.current_state = False
            logger.info("System reset to normal mode")
    
    def close(self):
        """Clean shutdown"""
        self.force_normal_mode()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logger.info("Serial connection closed")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    print("=" * 70)
    print("PROJECT MORPHOS - PHASE 4.0: ADVANCED FLASH DETECTION")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ YOLOv8 ambulance detection (97.4% mAP)")
    print("  ✓ HSV-based color analysis (lighting invariant)")
    print("  ✓ FFT frequency analysis (1-3 Hz emergency lights)")
    print("  ✓ Multiple ROI tracking (rotation resistant)")
    print("  ✓ Robust serial communication with auto-reconnect")
    print("  ✓ Fail-safe emergency protocols")
    print("  ✓ Manual 'C' key override for safety")
    print("=" * 70)
    
    # Initialize Arduino controller
    arduino = ArduinoController(Config.BAUD_RATE, Config.SERIAL_TIMEOUT)
    arduino.connect()
    
    # Initialize flash detector
    flash_detector = EmergencyFlashDetector(
        buffer_frames=Config.FLASH_BUFFER_SIZE,
        threshold_std=Config.FLASH_THRESHOLD_STD,
        target_frequency=Config.FLASH_FREQUENCY_RANGE,
        fps=Config.CAMERA_FPS
    )
    
    # Load YOLO model
    try:
        model_path = Config.get_model_path()
        logger.info(f"Loading model: {model_path}")
        # Use CUDA without FP16 (RTX 3060 stable)
        model = YOLO(model_path).to('cuda')
        logger.info("Model loaded on RTX 3060 (CUDA)")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        # Fallback to CPU if CUDA fails
        logger.warning(f"CUDA load failed ({e}), trying CPU")
        model = YOLO(model_path)
    
    # Initialize camera
    logger.info("Starting camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        sys.exit(1)
    
    # State variables
    consecutive_no_detection = 0
    consecutive_flash_frames = 0
    consecutive_no_flash = 0  # NEW: Track frames without flash
    flash_confirmed = False
    
    # Frame skipping optimization
    frame_counter = 0
    last_best_box = None
    last_best_conf = 0
    
    # Manual override flag
    manual_clear = False
    
    logger.info("System ready. Press 'Q' to quit, 'C' to force clear emergency.\n")
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            frame_counter += 1
            
            # YOLO inference every 2nd frame
            if frame_counter % 2 == 1:
                results = model.predict(frame, conf=Config.CONF_THRESHOLD, verbose=False)
                
                best_box = None
                best_conf = 0
                
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            best_box = tuple(map(int, box.xyxy[0]))
                
                last_best_box = best_box
                last_best_conf = best_conf
            else:
                best_box = last_best_box
                best_conf = last_best_conf
            
            # Process detection
            flash_detected = False
            flash_confidence = 0.0
            flash_color = "NONE"
            flash_frequency = 0.0
            
            if best_box is not None:
                consecutive_no_detection = 0
                
                # Analyze flash
                flash_detected, flash_confidence, flash_color, flash_frequency = \
                    flash_detector.update(frame, best_box)
                
                if flash_detected:
                    consecutive_flash_frames = min(consecutive_flash_frames + 1, Config.FLASH_CONFIRMATION_FRAMES * 2)
                    consecutive_no_flash = 0  # Reset no-flash counter
                else:
                    # Flash stopped - increment no-flash counter
                    consecutive_no_flash += 1
                    # Slow decay of flash frames
                    if frame_counter % 3 == 0:  # Decay every 3rd frame
                        consecutive_flash_frames = max(0, consecutive_flash_frames - 1)
                
                # Visualize detection
                x1, y1, x2, y2 = best_box
                
                # Color coding based on state
                if flash_confirmed:
                    box_color = (0, 0, 255)  # RED = Emergency active
                    status_label = f"EMERGENCY {flash_color}"
                elif consecutive_flash_frames > 0:
                    box_color = (0, 165, 255)  # ORANGE = Flashing detected
                    status_label = f"FLASHING ({consecutive_flash_frames}/{Config.FLASH_CONFIRMATION_FRAMES})"
                else:
                    box_color = (0, 255, 0)  # GREEN = Normal detection
                    status_label = f"AMBULANCE {best_conf:.0%}"
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, status_label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # Flash info
                if flash_detected or consecutive_flash_frames > 0:
                    cv2.putText(frame, f"Freq: {flash_frequency:.2f}Hz", (x1, y1-35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"NoFlash: {consecutive_no_flash}", (x1, y1-55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            else:
                # No detection at all
                consecutive_no_detection += 1
                consecutive_no_flash += 1
            
            # ================================================================
            # TRAFFIC CONTROL LOGIC (CORRECTED)
            # ================================================================
            
            # 1. TRIGGER: Detection + Flash confirmed for 1.5s
            if (best_box is not None and 
                consecutive_flash_frames >= Config.FLASH_CONFIRMATION_FRAMES and 
                not flash_confirmed):
                
                flash_confirmed = True
                arduino.activate_emergency()
                logger.info("🚨 EMERGENCY TRIGGERED")
            
            # 2. CLEAR CONDITIONS: Either no car OR flash lost for 2s
            # Condition A: No detection for 1 second
            elif consecutive_no_detection >= Config.FRAMES_TO_CLEAR:
                if flash_confirmed:
                    arduino.clear_emergency()
                    flash_confirmed = False
                    logger.info("✓ Emergency cleared - No car detected")
                
                # Reset all buffers
                flash_detector.reset()
                consecutive_flash_frames = 0
                consecutive_no_flash = 0
            
            # Condition B: Flash lost for 2 seconds (but car still visible)
            elif (flash_confirmed and 
                  consecutive_no_flash >= Config.FLASH_LOST_FRAMES):
                
                arduino.clear_emergency()
                flash_confirmed = False
                logger.info("✓ Emergency cleared - Flash stopped")
                
                # Keep car detection active but reset flash state
                consecutive_flash_frames = 0
                consecutive_no_flash = 0
            
            # 3. MANUAL OVERRIDE (from key press)
            if manual_clear:
                if flash_confirmed:
                    arduino.clear_emergency()
                flash_confirmed = False
                flash_detector.reset()
                consecutive_flash_frames = 0
                consecutive_no_flash = 0
                manual_clear = False
                logger.info("✓ Emergency cleared - MANUAL OVERRIDE")
            
            # ================================================================
            # UI OVERLAY
            # ================================================================
            # Emergency status banner
            if arduino.current_state:
                cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 255), -1)
                cv2.putText(frame, "EMERGENCY MODE", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                cv2.rectangle(frame, (10, 10), (400, 70), (0, 128, 0), -1)
                cv2.putText(frame, "NORMAL OPERATION", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Flash analysis info
            info_y = 100
            cv2.putText(frame, f"Flash Conf: {flash_confidence:.1f}%", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Color: {flash_color}", 
                       (10, info_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Freq: {flash_frequency:.2f}Hz", 
                       (10, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"NoFlash: {consecutive_no_flash}/{Config.FLASH_LOST_FRAMES}", 
                       (10, info_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 128), 1)
            
            # Serial status
            serial_status = "SERIAL: ✓" if arduino.is_connected else "SERIAL: ✗ (DEBUG)"
            serial_color = (0, 255, 0) if arduino.is_connected else (0, 0, 255)
            cv2.putText(frame, serial_status, (10, frame.shape[0]-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, serial_color, 2)
            
            # Controls hint
            cv2.putText(frame, "Q=Quit  C=Clear Emergency", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display
            cv2.imshow("Morphos Phase 4.0 - Emergency Clear Fixed", frame)
            
            # KEY HANDLING
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Shutdown requested by user")
                break
            elif key == ord('c'):
                manual_clear = True
                logger.info("Manual clear requested")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    finally:
        # CLEANUP
        logger.info("Shutting down...")
        arduino.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("System offline")

if __name__ == "__main__":
    main()