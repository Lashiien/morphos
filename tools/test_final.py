import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/morphos_training/exp_fixed/weights/best.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

print("OPTIMAL SETTINGS: conf=0.5")
print("Best performance under training room light")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # OPTIMAL: 50% confidence threshold
    results = model.predict(frame, conf=0.5, verbose=False)
    
    det_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            det_count += 1
            
            # Color coding by confidence
            if conf >= 0.7:
                color = (0, 255, 0)  # Green (excellent)
            elif conf >= 0.5:
                color = (0, 255, 255)  # Yellow (good)
            else:
                color = (0, 0, 255)  # Red (shouldn't show with conf=0.5)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"Ambulance {conf:.0%}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Status
    status = f"Detections: {det_count} | Threshold: 50%"
    cv2.putText(frame, status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Morphos Final Test (50% threshold)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()