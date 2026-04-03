from ultralytics import YOLO

def main():
    # Load fresh YOLOv8n (not your old weights)
    model = YOLO('yolov8n.pt')
    
    print("Starting training on new 300-image dataset...")
    
    # Train with conservative settings (small dataset)
    results = model.train(
        data='dataset_v2/Ambulance.v1i.yolov8/data.yaml',
        epochs=100,           # More epochs for small dataset
        imgsz=640,
        batch=8,             # Small batch for stability
        workers=0,           # Avoid multiprocessing errors
        device=0,            # Your RTX 3060
        patience=20,         # Early stopping if no improvement
        save=True,
        project='runs/detect/new_training',
        name='morphos_v2',
        pretrained=True,
        optimizer='AdamW',   # Better for small datasets
        lr0=0.001,           # Conservative learning rate
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,         # Less augmentation (small dataset)
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,        # Less rotation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment'
    )
    
    print(f"Training complete!")
    print(f"Best model saved to: runs/detect/new_training/morphos_v2/weights/best.pt")
    print(f"mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")

if __name__ == "__main__":
    main()