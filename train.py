"""
PROJECT MORPHOS - Training Script
Trains YOLOv8n model on ambulance detection dataset

Usage:
    python train.py

Requirements:
    - Images in data/raw/
    - Labels in data/labels/ (YOLO format .txt files)
    - Class: ambulance (class_id = 0)

Output:
    - Model weights saved to models/trained/best.pt
    - Training results printed to console
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from datetime import datetime

# Set random seeds for reproducibility
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
RAW_DATA_DIR = Path("data/raw")
LABELS_DATA_DIR = Path("data/labels")
OUTPUT_MODEL_DIR = Path("models/trained")
TEMP_DATASET_DIR = Path("data/morphos_training")

TRAIN_SPLIT = 0.8  # 80% train, 20% validation
CLASS_NAME = "ambulance"
CLASS_ID = 0

# Training hyperparameters
MODEL_VARIANT = "yolov8n.pt"  # nano - fast training
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640


def print_header(message: str):
    """Print a formatted header message."""
    print("\n" + "=" * 70)
    print(f"  {message}")
    print("=" * 70)


def print_step(message: str):
    """Print a step message."""
    print(f"\n>>> {message}")


def create_yolo_dataset_structure():
    """
    Create the YOLO-compatible dataset directory structure.
    Structure:
        data/morphos_training/
            images/
                train/
                val/
            labels/
                train/
                val/
    """
    print_step("Creating YOLO dataset directory structure...")
    
    # Clean up existing temp directory if it exists
    if TEMP_DATASET_DIR.exists():
        print(f"  Removing existing temp directory: {TEMP_DATASET_DIR}")
        shutil.rmtree(TEMP_DATASET_DIR)
    
    # Create directories
    (TEMP_DATASET_DIR / "images" / "train").mkdir(parents=True)
    (TEMP_DATASET_DIR / "images" / "val").mkdir(parents=True)
    (TEMP_DATASET_DIR / "labels" / "train").mkdir(parents=True)
    (TEMP_DATASET_DIR / "labels" / "val").mkdir(parents=True)
    
    print(f"  Created: {TEMP_DATASET_DIR}")
    return TEMP_DATASET_DIR


def find_images_and_labels():
    """Find all images in data/raw and their corresponding labels."""
    print_step("Scanning for images and labels...")
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    images = []
    for ext in image_extensions:
        images.extend(list(RAW_DATA_DIR.glob(f"*{ext}")))
        images.extend(list(RAW_DATA_DIR.glob(f"*{ext.upper()}")))
    
    images = sorted(set(images))  # Remove duplicates
    
    if not images:
        raise ValueError(f"No images found in {RAW_DATA_DIR}")
    
    print(f"  Found {len(images)} images in {RAW_DATA_DIR}")
    
    # Check for corresponding labels
    images_with_labels = []
    images_without_labels = []
    
    for img_path in images:
        label_path = LABELS_DATA_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            images_with_labels.append((img_path, label_path))
        else:
            images_without_labels.append(img_path)
    
    print(f"  Images with labels: {len(images_with_labels)}")
    print(f"  Images without labels: {len(images_without_labels)}")
    
    if images_without_labels:
        print(f"  WARNING: {len(images_without_labels)} images have no corresponding label file!")
        print(f"  These will NOT be included in training.")
        for img in images_without_labels[:5]:
            print(f"    - {img.name}")
        if len(images_without_labels) > 5:
            print(f"    ... and {len(images_without_labels) - 5} more")
    
    if len(images_with_labels) < 10:
        raise ValueError(
            f"Need at least 10 labeled images for training, but only found {len(images_with_labels)}. "
            f"Please label more images using label_tool.py"
        )
    
    return images_with_labels


def split_dataset(image_label_pairs):
    """Split dataset into train and validation sets (80/20)."""
    print_step("Splitting dataset into train/validation sets...")
    
    random.shuffle(image_label_pairs)
    
    split_index = int(len(image_label_pairs) * TRAIN_SPLIT)
    train_pairs = image_label_pairs[:split_index]
    val_pairs = image_label_pairs[split_index:]
    
    print(f"  Training set: {len(train_pairs)} images")
    print(f"  Validation set: {len(val_pairs)} images")
    
    return train_pairs, val_pairs


def copy_to_yolo_structure(pairs, split_name: str):
    """Copy images and labels to the YOLO directory structure."""
    print_step(f"Copying {split_name} data to YOLO structure...")
    
    images_dir = TEMP_DATASET_DIR / "images" / split_name
    labels_dir = TEMP_DATASET_DIR / "labels" / split_name
    
    for i, (img_path, label_path) in enumerate(pairs):
        # Copy image
        new_img_path = images_dir / img_path.name
        shutil.copy2(img_path, new_img_path)
        
        # Copy label
        new_label_path = labels_dir / label_path.name
        shutil.copy2(label_path, new_label_path)
        
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(pairs)} files...")
    
    print(f"  Copied {len(pairs)} {split_name} images/labels")


def create_data_yaml():
    """Create the data.yaml configuration file for YOLO training."""
    print_step("Creating data.yaml configuration...")
    
    data_yaml = {
        'path': str(TEMP_DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {CLASS_ID: CLASS_NAME}
    }
    
    yaml_path = TEMP_DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Created: {yaml_path}")
    print(f"  Content:")
    print(f"    path: {data_yaml['path']}")
    print(f"    train: {data_yaml['train']}")
    print(f"    val: {data_yaml['val']}")
    print(f"    nc: {data_yaml['nc']}")
    print(f"    names: {data_yaml['names']}")
    
    return yaml_path


def train_model(yaml_path):
    """Train the YOLOv8 model."""
    print_header("STARTING YOLOV8 TRAINING")
    
    from ultralytics import YOLO
    
    print_step(f"Loading base model: {MODEL_VARIANT}")
    model = YOLO(MODEL_VARIANT)
    
    print(f"  Model: {MODEL_VARIANT}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Data config: {yaml_path}")
    
    print_step("Training started (this may take 10-60 minutes depending on GPU)...")
    print("  Progress will be shown below as epochs complete.\n")
    
    # Train the model
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=str(TEMP_DATASET_DIR),
        name='train_output',
        exist_ok=True,
        verbose=True,
        patience=20,  # Early stopping after 20 epochs without improvement
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
    )
    
    return model, results


def copy_best_model():
    """Copy the best model weights to the output directory."""
    print_step("Finding and copying best model weights...")
    
    # The best weights are saved in the train_output/weights/ directory
    weights_dir = TEMP_DATASET_DIR / "train_output" / "weights"
    best_weights_path = weights_dir / "best.pt"
    last_weights_path = weights_dir / "last.pt"
    
    if best_weights_path.exists():
        print(f"  Found best.pt at: {best_weights_path}")
        
        # Ensure output directory exists
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Copy to output location
        output_path = OUTPUT_MODEL_DIR / "best.pt"
        shutil.copy2(best_weights_path, output_path)
        print(f"  Copied to: {output_path}")
        
        return output_path
    elif last_weights_path.exists():
        print(f"  WARNING: best.pt not found, using last.pt")
        print(f"  Found last.pt at: {last_weights_path}")
        
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        output_path = OUTPUT_MODEL_DIR / "best.pt"
        shutil.copy2(last_weights_path, output_path)
        print(f"  Copied to: {output_path}")
        
        return output_path
    else:
        raise FileNotFoundError(
            f"Could not find model weights in {weights_dir}. "
            f"Training may have failed."
        )


def print_training_summary(results_path):
    """Print training results summary."""
    print_header("TRAINING COMPLETE - RESULTS SUMMARY")
    
    print(f"\nOutput model location: {OUTPUT_MODEL_DIR / 'best.pt'}")
    print(f"\nTo use the model:")
    print(f"  1. The weights have been copied to: models/trained/best.pt")
    print(f"  2. Run: python morphos_final_v2.py")
    print(f"\nTemp training data is preserved at: {TEMP_DATASET_DIR}")
    print(f"  You can delete it with: shutil.rmtree('{TEMP_DATASET_DIR}')")
    
    # Try to load and display results if available
    try:
        from ultralytics import YOLO
        best_model = YOLO(str(OUTPUT_MODEL_DIR / "best.pt"))
        print("\n" + "=" * 70)
        print("  MODEL VALIDATION METRICS (from final validation run)")
        print("=" * 70)
        print("\n  To get detailed metrics, run validation separately:")
        print("  >>> from ultralytics import YOLO")
        print("  >>> model = YOLO('models/trained/best.pt')")
        print("  >>> results = model.val(data='data/morphos_training/data.yaml')")
        print("  >>> print(results.box.map)  # mAP50-95")
        print("  >>> print(results.box.map50)  # mAP50")
        print("  >>> print(results.box.map75)  # mAP75)")
    except Exception as e:
        print(f"\nNote: Could not load model for summary: {e}")


def cleanup_temp_files():
    """Prompt user to clean up temp files."""
    print("\n" + "=" * 70)
    print("  CLEANUP OPTION")
    print("=" * 70)
    response = input(f"\nDelete temp training data at {TEMP_DATASET_DIR}? (y/n): ").strip().lower()
    if response == 'y':
        shutil.rmtree(TEMP_DATASET_DIR)
        print(f"Deleted: {TEMP_DATASET_DIR}")
    else:
        print(f"Kept: {TEMP_DATASET_DIR}")


def main():
    """Main training pipeline."""
    print_header("MORPHOS YOLOV8 TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_VARIANT}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Train/Val split: {int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}")
    print(f"  Source images: {RAW_DATA_DIR}")
    print(f"  Source labels: {LABELS_DATA_DIR}")
    print(f"  Output model: {OUTPUT_MODEL_DIR / 'best.pt'}")
    
    try:
        # Step 1: Create YOLO dataset structure
        create_yolo_dataset_structure()
        
        # Step 2: Find images and labels
        image_label_pairs = find_images_and_labels()
        
        # Step 3: Split dataset
        train_pairs, val_pairs = split_dataset(image_label_pairs)
        
        # Step 4: Copy files to YOLO structure
        copy_to_yolo_structure(train_pairs, "train")
        copy_to_yolo_structure(val_pairs, "val")
        
        # Step 5: Create data.yaml
        yaml_path = create_data_yaml()
        
        # Step 6: Train model
        model, results = train_model(yaml_path)
        
        # Step 7: Copy best model to output location
        output_path = copy_best_model()
        
        # Step 8: Print summary
        print_training_summary(output_path)
        
        # Step 9: Cleanup option
        cleanup_temp_files()
        
        print_header("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("  TRAINING FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())