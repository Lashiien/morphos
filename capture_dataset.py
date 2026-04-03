"""
Professional Dataset Capture Script for Ambulance Detection Training

A clean, dumb camera capture utility for collecting training images.
NO AI logic - purely for data collection.

Controls:
    SPACEBAR - Capture image
    Q        - Quit

Output:
    Saves images to /data/raw/ directory
    Naming convention: ambulance_<YYYYMMDD_HHMMSS>_<Counter>.jpg
"""

import cv2
import os
from datetime import datetime
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = Path("data/raw")
WINDOW_NAME = "Morphos Dataset Capture"
FEEDBACK_DURATION_MS = 500  # 0.5 seconds


def ensure_output_directory() -> Path:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def generate_filename(counter: int) -> str:
    """
    Generate filename with timestamp and counter.
    Format: ambulance_<YYYYMMDD_HHMMSS>_<Counter>.jpg
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ambulance_{timestamp}_{counter:03d}.jpg"


def draw_overlay(frame, images_captured: int, feedback_message: str = None):
    """
    Draw UI overlay on the video frame.
    
    Args:
        frame: The video frame to annotate
        images_captured: Number of images captured so far
        feedback_message: Optional feedback message (e.g., "SAVED!")
    """
    display = frame.copy()
    
    # Semi-transparent background for text readability
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (350, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    
    # Main counter display
    cv2.putText(
        display,
        f"Images Captured: {images_captured}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # Controls hint
    cv2.putText(
        display,
        "SPACE=Capture | Q=Quit",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1
    )
    
    # Feedback message (e.g., "SAVED!")
    if feedback_message:
        # Large centered feedback
        text_size = cv2.getTextSize(feedback_message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (display.shape[1] - text_size[0]) // 2
        text_y = (display.shape[0] + text_size[1]) // 2
        
        # Green flash background
        flash_overlay = display.copy()
        cv2.rectangle(
            flash_overlay,
            (text_x - 20, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 20, text_y + 10),
            (0, 200, 0),
            -1
        )
        cv2.addWeighted(flash_overlay, 0.7, display, 0.3, 0, display)
        
        cv2.putText(
            display,
            feedback_message,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3
        )
    
    return display


def main():
    """Main capture loop."""
    # Ensure output directory exists
    output_path = ensure_output_directory()
    print(f"Output directory: {output_path.absolute()}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "=" * 60)
    print("MORPHOS DATASET CAPTURE")
    print("=" * 60)
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Output: {output_path.absolute()}")
    print("\nCONTROLS:")
    print("  [SPACEBAR] = Capture image")
    print("  [Q]        = Quit")
    print("=" * 60 + "\n")
    
    # State variables
    counter = 0
    feedback_message = None
    feedback_end_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera!")
                break
            
            # Check if feedback message should still be displayed
            current_time = cv2.getTickCount()
            if feedback_message and current_time > feedback_end_time:
                feedback_message = None
            
            # Draw overlay
            display = draw_overlay(frame, counter, feedback_message)
            
            # Show live feed
            cv2.imshow(WINDOW_NAME, display)
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACEBAR - capture
                filename = generate_filename(counter)
                filepath = output_path / filename
                
                # Save original frame without overlay
                cv2.imwrite(str(filepath), frame)
                
                # Set feedback
                feedback_message = "SAVED!"
                feedback_end_time = current_time + int(FEEDBACK_DURATION_MS * cv2.getTickFrequency() / 1000)
                
                counter += 1
                print(f"✓ Saved: {filename}")
                
            elif key == ord('q') or key == ord('Q'):  # QUIT
                print(f"\nExiting. Total images captured: {counter}")
                break
    
    finally:
        # Clean exit - release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Windows destroyed.")
    
    print("\n" + "=" * 60)
    print(f"CAPTURE COMPLETE")
    print(f"Total images: {counter}")
    print(f"Saved to: {output_path.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()