from ultralytics import YOLO
import torch

def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU will be slow.")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='morphos_data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=10,
        project='morphos_training',
        name='exp',
        exist_ok=True,
    )
    
    print(f"\nTraining complete! Best model: {results.best}")

if __name__ == '__main__':
    main()