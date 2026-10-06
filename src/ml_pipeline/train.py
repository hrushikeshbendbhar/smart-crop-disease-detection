from pathlib import Path
from ultralytics import YOLO

def train_crop_disease_model_cpu_max():
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = (project_root / "data" / "raw" / "PlantVillage").as_posix()
    models_dir = (project_root / "models").as_posix()
    
    print("⚡ Loading YOLOv8 Nano classification model...")
    model = YOLO("yolov8n-cls.pt") 
    
    print("🚀 Starting max-performance CPU training (90%+ CPU Target)...")
    
    results = model.train(
        data=dataset_path,
        epochs=15,             # 15 epochs
        imgsz=224,             # Fast 224x224 resolution
        batch=64,              # Large batch size to keep all CPU cores busy
        workers=12,            # Uses all 12 threads of i5-12400 for max CPU load
        cache=True,            # Caches dataset into RAM for max throughput
        project=models_dir,    
        name="disease_classifier_v2",
        exist_ok=True,
        # Real-world augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5
    )
    
    print("\n✅ Training Complete!")

if __name__ == "__main__":
    train_crop_disease_model_cpu_max()