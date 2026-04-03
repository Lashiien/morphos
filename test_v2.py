# test_v2.py
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/runs/detect/new_training/morphos_v2/weights/best.pt')
print("Model loaded: 97.4% mAP")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model(frame, conf=0.5)
    annotated = results[0].plot()
    
    cv2.imshow("Morphos v2 - 97.4% mAP Model", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()