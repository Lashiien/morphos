import cv2
import os
import glob

# CONFIGURATION
IMAGE_DIR = "data/raw"
OUTPUT_DIR = "data/labels"
CLASS_NAME = "ambulance"
CLASS_ID = 0

# Create directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all images
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + 
                glob.glob(os.path.join(IMAGE_DIR, "*.png")))
current = 0
total = len(images)

# Drawing state
drawing = False
ix, iy = -1, -1
boxes = []  # Store boxes for current image

def normalize(x, y, w, h, img_w, img_h):
    """Convert to YOLO format (0-1)"""
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height

def draw_boxes(img, boxes):
    """Draw existing boxes"""
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, boxes, img_display, img_orig
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_orig.copy()
            img_display = draw_boxes(img_display, boxes)
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        w, h = x2 - x1, y2 - y1
        
        if w > 20 and h > 20:  # Minimum size
            boxes.append((x1, y1, w, h))
            img_display = img_orig.copy()
            img_display = draw_boxes(img_display, boxes)

print(f"Found {total} images to label")
print("Controls:")
print("  SPACE = Save labels & Next Image")
print("  Z = Undo last box")
print("  N = Skip image (no labels)")
print("  Q = Quit")

while current < total:
    img_path = images[current]
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        current += 1
        continue
        
    img_h, img_w = img_orig.shape[:2]
    img_display = img_orig.copy()
    boxes = []  # Reset boxes for this image
    
    # Check if already labeled (allow reviewing)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(IMAGE_DIR, "..", "labels", f"{base_name}.txt")
    if os.path.exists(label_path):
        print(f"Already labeled: {base_name}.txt exists")

    cv2.namedWindow(f"Labeling: {current+1}/{total} - {base_name}")
    cv2.setMouseCallback(f"Labeling: {current+1}/{total} - {base_name}", mouse_callback)
    
    while True:
        cv2.imshow(f"Labeling: {current+1}/{total} - {base_name}", img_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # SPACE: Save and next
            # Save YOLO format labels
            if len(boxes) > 0:
                label_file = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
                with open(label_file, 'w') as f:
                    for (x, y, w, h) in boxes:
                        xn, yn, wn, hn = normalize(x, y, w, h, img_w, img_h)
                        f.write(f"{CLASS_ID} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")
                print(f"Saved: {label_file} ({len(boxes)} boxes)")
            else:
                print(f"Skipped: {base_name} (no boxes)")
                
            current += 1
            break
            
        elif key == ord('z'):  # Z: Undo last box
            if len(boxes) > 0:
                boxes.pop()
                img_display = img_orig.copy()
                img_display = draw_boxes(img_display, boxes)
                print("Undo last box")
                
        elif key == ord('n'):  # N: Skip image
            print(f"Skipped: {base_name}")
            current += 1
            break
            
        elif key == ord('q'):  # Q: Quit
            print("Quitting... Progress saved.")
            current = total
            break

cv2.destroyAllWindows()
print(f"\nLabeling complete!")
print(f"Labels saved to: {OUTPUT_DIR}")
print(f"Check files: {len(os.listdir(OUTPUT_DIR))} label files created")