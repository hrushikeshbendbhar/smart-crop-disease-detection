import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Resolve base path dynamically relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "disease_classifier_v2" / "weights" / "best.pt"

class DiseaseClassifierEngine:
    def __init__(self, model_path: str = str(MODEL_PATH), conf_threshold: float = 0.65):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"📦 Loading Vision Engine Weights from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def preprocess_image(self, image_input) -> np.ndarray:
        """Preprocesses real-world images: crops leaf area to mitigate background noise."""
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Unsupported image format.")

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Mask for foliage ranges (green/brown)
        lower_plant = np.array([10, 20, 20])
        upper_plant = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_plant, upper_plant)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                pad = 15
                h_img, w_img, _ = image.shape
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
                return image[y1:y2, x1:x2]

        return image

    def predict(self, image_input):
        """Runs prediction with confidence safeguards against domain shift."""
        processed_image = self.preprocess_image(image_input)
        results = self.model.predict(source=processed_image, conf=0.25, verbose=False)
        
        result = results[0]
        top1_index = result.probs.top1
        class_name = result.names[top1_index]
        confidence = float(result.probs.top1conf.item())

        if confidence < self.conf_threshold:
            return {
                "status": "UNCERTAIN",
                "message": "Low confidence prediction (possible domain shift). Please upload a clearer photo of a single leaf.",
                "raw_prediction": class_name,
                "confidence": round(confidence * 100, 2)
            }

        formatted_name = class_name.replace("___", " - ").replace("_", " ")

        return {
            "status": "SUCCESS",
            "crop_disease": formatted_name,
            "raw_class": class_name,
            "confidence": round(confidence * 100, 2)
        }

if __name__ == "__main__":
    engine = DiseaseClassifierEngine()
    print("Engine initialized successfully!")