import cv2
import torch
import numpy as np
import os
import urllib.request
import time
import argparse

# Directory setup
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Only focus on road vehicles, with extended classifications
VEHICLE_CLASSES = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
# Additional vehicle types to detect (mapped to existing classes)
VEHICLE_MAPPING = {
    'car': ['car', 'van', 'sedan', 'suv', 'minivan'],
    'truck': ['truck', 'pickup', 'lorry'],
    'motorcycle': ['motorcycle', 'scooter', 'autorickshaw', 'tuk-tuk'],
    'bus': ['bus', 'minibus'],
    'bicycle': ['bicycle']
}

# Reverse mapping to display proper vehicle types
REVERSE_MAPPING = {}
for main_class, variants in VEHICLE_MAPPING.items():
    for variant in variants:
        REVERSE_MAPPING[variant] = main_class

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Vehicle Detection")
    parser.add_argument("--model", type=str, default="yolov5n", choices=["yolov5n", "yolov5s"],
                        help="Model to use: yolov5n (faster, less accurate) or yolov5s (slower, more accurate)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", 
                        help="Device to use: 'cpu', '0' (for CUDA), or leave empty for auto selection")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height") 
    parser.add_argument("--fps", action="store_true", help="Display FPS")
    parser.add_argument("--detailed", action="store_true", help="Use detailed vehicle classification")
    return parser.parse_args()

def download_yolov5_model(model_name="yolov5n"):
    """Download YOLOv5 model (n=nano variant for Raspberry Pi)"""
    model_path = f"{MODEL_DIR}/{model_name}.pt"
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model...")
        url = f"https://github.com/ultralytics/yolov5/releases/download/v6.0/{model_name}.pt"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    return model_path

def load_model(model_path, device="", conf=0.45):
    """Load YOLOv5 model with optimizations for Raspberry Pi"""
    try:
        # Auto-select device if not specified
        if not device:
            device = "cpu"  # Default to CPU for Raspberry Pi
        
        model_name = os.path.basename(model_path).split(".")[0] if model_path else "yolov5n"
        print(f"Loading model on device: {device}")
        
        if not model_path or not os.path.exists(model_path):
            print("Model path not found. Downloading from torch hub...")
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device)
        else:
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
            except Exception as e:
                print(f"Error loading model from file: {e}")
                print("Loading model from torch hub instead...")
                model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, device=device)
        
        # Optimization settings for Raspberry Pi
        model.conf = conf  # Confidence threshold
        model.iou = 0.45   # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        model.max_det = 100  # Maximum number of detections
        
        # Only detect vehicle classes (COCO class indices)
        model.classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def classify_vehicle_type(frame, bbox, class_name):
    """Further classify vehicle type based on features (shape, size, etc.)
    This is a basic implementation that can be enhanced with a dedicated classifier"""
    
    # Extract the vehicle image from the frame
    xmin, ymin, xmax, ymax = bbox
    vehicle_img = frame[ymin:ymax, xmin:xmax]
    
    if vehicle_img.size == 0:
        return class_name  # Return original class if crop failed
    
    # Simple rule-based classification
    aspect_ratio = (xmax - xmin) / (ymax - ymin)
    
    # Basic classification based on aspect ratio and size
    if class_name == 'car':
        if aspect_ratio > 1.5:  # Longer vehicles might be vans
            return 'van'
        else:
            return 'car'
    elif class_name == 'motorcycle':
        if aspect_ratio < 0.8 and (xmax - xmin) > 100:  # Wider than typical motorcycle
            return 'autorickshaw'
        else:
            return 'motorcycle'
    elif class_name == 'truck':
        if aspect_ratio > 2:  # Very long trucks
            return 'lorry'
        else:
            return 'truck'
    
    return class_name

def detect_vehicles(model, frame, detailed_classification=False):
    """Detect vehicles in a frame using YOLOv5"""
    if model is None:
        return pd.DataFrame()  # Return empty dataframe if model failed to load
        
    # Inference
    results = model(frame)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Filter to only include vehicle classes
    detections = detections[detections['name'].isin(VEHICLE_CLASSES)]
    
    # Apply detailed classification if enabled
    if detailed_classification and not detections.empty:
        detailed_classes = []
        for _, detection in detections.iterrows():
            bbox = (int(detection['xmin']), int(detection['ymin']), 
                   int(detection['xmax']), int(detection['ymax']))
            detailed_class = classify_vehicle_type(
                frame, bbox, detection['name'])
            detailed_classes.append(detailed_class)
        
        # Add the detailed classes as a new column
        detections['detailed_class'] = detailed_classes
    else:
        detections['detailed_class'] = detections['name']
    
    return detections

def count_vehicles(detections):
    """Count vehicles by type"""
    counts = {}
    
    # Use detailed_class column if available, otherwise use name
    class_column = 'detailed_class' if 'detailed_class' in detections.columns else 'name'
    
    for _, detection in detections.iterrows():
        class_name = detection[class_column]
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1
    return counts

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Download and load model
    print("Setting up vehicle detection model...")
    model_path = download_yolov5_model(args.model)
    model = load_model(model_path, device=args.device, conf=args.conf)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print("Model loaded successfully!")
    print(f"Detailed classification: {'Enabled' if args.detailed else 'Disabled'}")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set capture properties to optimize for Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Press 'q' to quit the application")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break
        
        frame_count += 1
        
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time
            
        # Detect vehicles with detailed classification if enabled
        detections = detect_vehicles(model, frame, args.detailed)
        
        # Count vehicles by type
        vehicle_counts = count_vehicles(detections)
        
        # Draw detections
        for _, detection in detections.iterrows():
            # Extract detection info
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = float(detection['confidence'])
            
            # Use detailed class if available
            class_name = detection['detailed_class'] if 'detailed_class' in detection else detection['name']
            base_class = detection['name']
            
            # Color based on vehicle type (using base class for color consistency)
            color = (0, 255, 0)  # Default green
            if base_class == 'car':
                color = (0, 255, 0)  # Green
            elif base_class == 'truck':
                color = (0, 0, 255)  # Red
            elif base_class == 'bus':
                color = (255, 0, 0)  # Blue
            elif base_class == 'motorcycle':
                color = (255, 255, 0)  # Cyan
            elif base_class == 'bicycle':
                color = (0, 255, 255)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw vehicle counts
        y_pos = 30
        for vehicle_type, count in vehicle_counts.items():
            cv2.putText(frame, f"{vehicle_type}: {count}", (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
        
        # Display FPS if enabled
        if args.fps:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Vehicle Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # On Raspberry Pi, make sure to import pandas here to avoid slow startup
    import pandas as pd
    main()