import cv2
import numpy as np
from collections import deque
from typing import Tuple, List, Optional
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergencyFlashDetector:
    """
    Advanced emergency light detector with:
    - HSV color space analysis (lighting invariance)
    - Frequency analysis (1-3 Hz emergency flash detection)
    - Multiple ROI tracking (handles rotation/cropping)
    - Temporal filtering
    """
    
    def __init__(
        self, 
        buffer_frames: int = 30,  # 1 sec @ 30fps
        threshold_std: float = 25.0,
        roi_resize: Tuple[int, int] = (48, 48),
        target_frequency: Tuple[float, float] = (1.0, 3.0),  # 1-3 Hz
        fps: float = 30.0
    ):
        """
        Args:
            buffer_frames: Number of frames to analyze (temporal window)
            threshold_std: Std dev threshold for color variance
            roi_resize: Resize ROI for faster processing
            target_frequency: Emergency flash frequency range (Hz)
            fps: Camera frame rate
        """
        # Color channel buffers (HSV space)
        self.hue_red_buffer = deque(maxlen=buffer_frames)
        self.hue_blue_buffer = deque(maxlen=buffer_frames)
        self.saturation_buffer = deque(maxlen=buffer_frames)
        self.value_buffer = deque(maxlen=buffer_frames)
        
        # Multiple ROI buffers (center, top-left, top-right)
        self.roi_buffers = {
            'center': deque(maxlen=buffer_frames),
            'top_left': deque(maxlen=buffer_frames),
            'top_right': deque(maxlen=buffer_frames)
        }
        
        # Motion stability tracking
        self.last_bbox_center = None
        self.motion_threshold = 40  # pixels
        
        # Color-specific pixel count buffers for red/blue alternation detection
        self.red_pixel_buffer = deque(maxlen=buffer_frames)
        self.blue_pixel_buffer = deque(maxlen=buffer_frames)
        
        self.threshold = threshold_std
        self.roi_size = roi_resize
        self.target_freq_range = target_frequency
        self.fps = fps
        self.buffer_size = buffer_frames
        
        # Frequency analysis setup
        self.min_freq, self.max_freq = target_frequency
        
    def extract_multiple_rois(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int]
    ) -> dict:
        """
        Extract multiple regions of interest to handle rotated/cropped ambulances
        Returns: dict of ROI regions
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = frame.shape[:2]
        
        # Clamp bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return {}
        
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Safety check for dimensions
        if bbox_w <= 10 or bbox_h <= 10:
            return {}
        
        rois = {}
        
        # Center ROI (25% of bbox - where lights typically are)
        cx, cy = x1 + bbox_w // 2, y1 + bbox_h // 2
        crop_w = max(10, bbox_w // 4)
        crop_h = max(10, bbox_h // 4)
        
        center_x1 = max(0, cx - crop_w // 2)
        center_y1 = max(0, cy - crop_h // 2)
        center_x2 = min(w, cx + crop_w // 2)
        center_y2 = min(h, cy + crop_h // 2)
        
        if center_x2 > center_x1 and center_y2 > center_y1:
            rois['center'] = frame[center_y1:center_y2, center_x1:center_x2]
        
        # Top-left ROI (handles rotated ambulances)
        tl_size = max(10, min(bbox_w, bbox_h) // 5)
        tl_x2 = min(x2, x1 + tl_size)
        tl_y2 = min(y2, y1 + tl_size)
        if tl_x2 > x1 and tl_y2 > y1:
            rois['top_left'] = frame[y1:tl_y2, x1:tl_x2]
        
        # Top-right ROI
        tr_x1 = max(x1, x2 - tl_size)
        tr_y2 = min(y2, y1 + tl_size)
        if x2 > tr_x1 and tr_y2 > y1:
            rois['top_right'] = frame[y1:tr_y2, tr_x1:x2]
        
        return rois
    
    def analyze_hsv_channels(self, roi: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Analyze ROI in HSV color space for better lighting invariance
        Returns: (hue_red_score, hue_blue_score, saturation, value)
        """
        if roi.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        if self.roi_size:
            roi = cv2.resize(roi, self.roi_size)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Red hue detection (wraps around: 0-10 and 170-180 in OpenCV)
        red_mask1 = (h >= 0) & (h <= 10)
        red_mask2 = (h >= 170) & (h <= 180)
        red_mask = red_mask1 | red_mask2
        
        # Blue hue detection (100-130 in OpenCV)
        blue_mask = (h >= 100) & (h <= 130)
        
        # Calculate mean saturation and value (brightness)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        
        # Score red/blue presence (weighted by saturation)
        red_score = np.mean(s[red_mask]) if np.any(red_mask) else 0.0
        blue_score = np.mean(s[blue_mask]) if np.any(blue_mask) else 0.0
        
        return red_score, blue_score, mean_s, mean_v
    
    def count_color_pixels(self, roi: np.ndarray) -> Tuple[int, int]:
        """
        Count pixels in red and blue emergency color ranges.
        Uses stricter thresholds: H in range, S > 100, V > 100
        
        Returns: (red_pixel_count, blue_pixel_count)
        """
        if roi.size == 0:
            return 0, 0
        
        # Resize for faster processing
        if self.roi_size:
            roi = cv2.resize(roi, self.roi_size)
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Red pixels: H in 0-10 or 170-180, S > 100, V > 100
        red_mask1 = (h >= 0) & (h <= 10) & (s >= 100) & (v >= 100)
        red_mask2 = (h >= 170) & (h <= 180) & (s >= 100) & (v >= 100)
        red_mask = red_mask1 | red_mask2
        red_count = np.count_nonzero(red_mask)
        
        # Blue pixels: H in 100-130, S > 100, V > 100
        blue_mask = (h >= 100) & (h <= 130) & (s >= 100) & (v >= 100)
        blue_count = np.count_nonzero(blue_mask)
        
        return red_count, blue_count
    
    def check_color_alternation(self) -> float:
        """
        Check if red and blue pixel counts are alternating over time.
        Returns a boost factor (1.0 = no boost, up to 1.5 = strong alternation).
        """
        if len(self.red_pixel_buffer) < self.buffer_size // 2:
            return 1.0
        
        red_arr = np.array(self.red_pixel_buffer)
        blue_arr = np.array(self.blue_pixel_buffer)
        
        # Check for negative correlation (alternation)
        if np.std(red_arr) < 1 or np.std(blue_arr) < 1:
            return 1.0
        
        correlation = np.corrcoef(red_arr, blue_arr)[0, 1]
        
        # Strong negative correlation = alternating pattern
        if correlation < -0.3:
            # Boost factor based on correlation strength
            boost = 1.0 + (abs(correlation) * 0.5)  # Up to 1.5x boost
            return min(1.5, boost)
        
        return 1.0
    
    def analyze_frequency(self, signal_buffer: List[float]) -> Tuple[bool, float]:
        """
        FFT-based frequency analysis to detect 1-3 Hz flashing
        Returns: (is_in_emergency_range, dominant_frequency)
        """
        if len(signal_buffer) < self.buffer_size // 2:
            return False, 0.0
        
        signal_array = np.array(signal_buffer)
        
        # Remove DC offset
        signal_array = signal_array - np.mean(signal_array)
        
        # Apply Hamming window to reduce spectral leakage
        windowed = signal_array * np.hamming(len(signal_array))
        
        # Compute FFT
        fft = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(len(signal_array), 1.0 / self.fps)
        
        # Only analyze positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Find dominant frequency (exclude DC component)
        if len(magnitude) > 1:
            dominant_idx = np.argmax(magnitude[1:]) + 1
            dominant_freq = abs(positive_freqs[dominant_idx])
            
            # Check if in emergency range (1-3 Hz)
            in_range = self.min_freq <= dominant_freq <= self.max_freq
            
            return in_range, dominant_freq
        
        return False, 0.0
    
    def update(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, float, str, float]:
        """
        Main update method with comprehensive flash detection
        
        Returns:
            (is_flashing, confidence_score, color_detected, frequency)
        """
        # ========================================================================
        # MOTION STABILITY CHECK
        # ========================================================================
        bbox_center_x = (bbox[0] + bbox[2]) // 2
        bbox_center_y = (bbox[1] + bbox[3]) // 2
        current_center = (bbox_center_x, bbox_center_y)
        
        if self.last_bbox_center is not None:
            # Calculate pixel movement since last frame
            dx = current_center[0] - self.last_bbox_center[0]
            dy = current_center[1] - self.last_bbox_center[1]
            movement = np.sqrt(dx*dx + dy*dy)
            
            if movement > self.motion_threshold:
                # Motion too fast - skip flash detection but still update center
                logger.debug(f"Motion too fast - skipping flash analysis (moved {movement:.1f}px)")
                self.last_bbox_center = current_center
                # Return partial result without counting toward confirmation
                return False, 0.0, "NONE", 0.0
        
        self.last_bbox_center = current_center
        # ========================================================================
        
        rois = self.extract_multiple_rois(frame, bbox)
        
        if not rois:
            return False, 0.0, "NONE", 0.0
        
        # Analyze primary ROI (center)
        if 'center' not in rois:
            return False, 0.0, "NONE", 0.0
        
        red_score, blue_score, sat, val = self.analyze_hsv_channels(rois['center'])
        
        # Store in buffers
        self.hue_red_buffer.append(red_score)
        self.hue_blue_buffer.append(blue_score)
        self.saturation_buffer.append(sat)
        self.value_buffer.append(val)
        
        # ========================================================================
        # COLOR-SPECIFIC PIXEL COUNTING (for alternation detection)
        # ========================================================================
        red_pixel_count, blue_pixel_count = self.count_color_pixels(rois['center'])
        self.red_pixel_buffer.append(red_pixel_count)
        self.blue_pixel_buffer.append(blue_pixel_count)
        # ========================================================================
        
        # Also analyze secondary ROIs for robustness
        for roi_name, roi_img in rois.items():
            r, b, s, v = self.analyze_hsv_channels(roi_img)
            self.roi_buffers[roi_name].append(max(r, b))  # Store dominant color
        
        # Need full buffer for analysis
        if len(self.hue_red_buffer) < self.buffer_size:
            return False, 0.0, "NONE", 0.0
        
        # Calculate temporal variance (detects flashing)
        std_red = np.std(np.array(self.hue_red_buffer))
        std_blue = np.std(np.array(self.hue_blue_buffer))
        std_sat = np.std(np.array(self.saturation_buffer))
        std_val = np.std(np.array(self.value_buffer))
        
        # Frequency analysis on combined signal
        combined_signal = list(self.hue_red_buffer) if std_red > std_blue else list(self.hue_blue_buffer)
        freq_match, dominant_freq = self.analyze_frequency(combined_signal)
        
        # Detection logic:
        # 1. High variance in RED or BLUE channel
        # 2. Variance significantly higher than ambient (saturation/value)
        # 3. Frequency in 1-6 Hz range (emergency lights + harmonics)
        
        is_red_flash = (std_red > self.threshold) and (std_red > std_sat * 1.3)
        is_blue_flash = (std_blue > self.threshold) and (std_blue > std_sat * 1.3)
        
        # Confidence score (0-100)
        confidence = 0.0
        color = "NONE"
        
        if (is_red_flash or is_blue_flash) and freq_match:
            if is_red_flash and is_blue_flash:
                color = "RED_BLUE"
                confidence = min(100, (max(std_red, std_blue) / self.threshold) * 50)
            elif is_red_flash:
                color = "RED"
                confidence = min(100, (std_red / self.threshold) * 50)
            elif is_blue_flash:
                color = "BLUE"
                confidence = min(100, (std_blue / self.threshold) * 50)
            
            # Boost confidence if frequency is strong match
            if freq_match:
                confidence = min(100, confidence * 1.5)
            
            # ====================================================================
            # SECONDARY SIGNAL: Color alternation boost (red/blue pixels)
            # This is additive - treats alternation as bonus evidence
            # ====================================================================
            alternation_boost = self.check_color_alternation()
            if alternation_boost > 1.0:
                confidence = min(100, confidence * alternation_boost)
                logger.debug(f"Color alternation boost applied: {alternation_boost:.2f}x")
            # ====================================================================
            
            return True, confidence, color, dominant_freq
        
        return False, max(std_red, std_blue), "NONE", dominant_freq
    
    def reset(self):
        """Clear all buffers when object is lost"""
        self.hue_red_buffer.clear()
        self.hue_blue_buffer.clear()
        self.saturation_buffer.clear()
        self.value_buffer.clear()
        self.red_pixel_buffer.clear()
        self.blue_pixel_buffer.clear()
        for buf in self.roi_buffers.values():
            buf.clear()
        self.last_bbox_center = None
        logger.info("Flash detector buffers reset")


# STANDALONE TEST
if __name__ == "__main__":
    detector = EmergencyFlashDetector(
        buffer_frames=30, 
        threshold_std=20.0,
        target_frequency=(1.0, 3.0),
        fps=30.0
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("=" * 60)
    print("ADVANCED EMERGENCY FLASH DETECTOR TEST")
    print("=" * 60)
    print("Features:")
    print("  • HSV color space analysis")
    print("  • 1-3 Hz frequency detection (FFT)")
    print("  • Multiple ROI tracking")
    print("  • Lighting invariance")
    print("\nPress 'Q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        bbox = (w//4, h//4, w*3//4, h*3//4)  # Center region
        
        is_flash, confidence, color, freq = detector.update(frame, bbox)
        
        # Visual feedback
        status_color = (0, 0, 255) if is_flash else (0, 255, 0)
        status_msg = f"{color} EMERGENCY" if is_flash else "NO EMERGENCY"
        
        # Draw bbox
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), status_color, 3)
        
        # Status display
        cv2.putText(frame, f"Status: {status_msg}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frequency: {freq:.2f} Hz", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Target range indicator
        freq_status = "✓ IN RANGE" if 1.0 <= freq <= 3.0 else "✗ OUT OF RANGE"
        cv2.putText(frame, f"1-3 Hz: {freq_status}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if 1.0 <= freq <= 3.0 else (0, 0, 255), 2)
        
        cv2.imshow("Emergency Flash Detector - HSV + FFT", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()