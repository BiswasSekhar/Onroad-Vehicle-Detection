import cv2
import torch
import numpy as np
import os
import urllib.request
import time
import argparse
import threading

# Directory setup
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Only focus on road vehicles
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
    parser = argparse.ArgumentParser(description="Vehicle Detection for Raspberry Pi")
    parser.add_argument("--model", type=str, default="yolov5n", choices=["yolov5n", "yolov5n6"],
                        help="Model to use: yolov5n (fastest) or yolov5n6 (slightly more accurate)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=416, help="Frame width (smaller is faster)")
    parser.add_argument("--height", type=int, default=416, help="Frame height (smaller is faster)") 
    parser.add_argument("--fps", action="store_true", help="Display FPS")
    parser.add_argument("--detailed", action="store_true", help="Use detailed vehicle classification")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (usually 0)")
    parser.add_argument("--skip-frames", type=int, default=2, 
                        help="Process every Nth frame for better performance (default: 2)")
    parser.add_argument("--headless", action="store_true", 
                        help="Run without display window (for headless Raspberry Pi)")
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

def load_model(model_path, conf=0.45):
    """Load YOLOv5 model with optimizations for Raspberry Pi"""
    try:
        # Always use CPU for Raspberry Pi
        device = "cpu"
        
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
        model.max_det = 50  # Maximum number of detections (reduced for Pi)
        
        # Only detect vehicle classes (COCO class indices)
        model.classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def classify_vehicle_type(frame, bbox, class_name):
    """Basic vehicle subtype classification based on shape/size"""
    # Extract the vehicle image from the frame
    xmin, ymin, xmax, ymax = bbox
    
    # Simple rule-based classification based on aspect ratio
    aspect_ratio = (xmax - xmin) / max(1, (ymax - ymin))  # Avoid division by zero
    area = (xmax - xmin) * (ymax - ymin)
    
    if class_name == 'car':
        if aspect_ratio > 1.5:  # Longer vehicles might be vans
            return 'van'
        else:
            return 'car'
    elif class_name == 'motorcycle':
        if aspect_ratio < 0.8 and area > 5000:  # Wider than typical motorcycle
            return 'autorickshaw'
        else:
            return 'motorcycle'
    elif class_name == 'truck':
        if aspect_ratio > 2:  # Very long trucks
            return 'lorry'
        else:
            return 'truck'
    
    return class_name

class VideoStreamThread:
    """Thread class for capturing video frames in a separate thread"""
    def __init__(self, src=0, width=416, height=416):
        self.src = src
        self.width = width
        self.height = height
        self.stopped = False
        self.frame = None
        
    def start(self):
        print("Starting video stream thread...")
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            return False
            
        # Start thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return True
        
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def detect_vehicles(model, frame, detailed_classification=False):
    """Detect vehicles in a frame using YOLOv5"""
    if model is None:
        return pd.DataFrame()  # Return empty dataframe if model failed to load
    
    try:    
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
    except Exception as e:
        print(f"Error in detection: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

def draw_detections(frame, detections, show_fps=False, fps=0):
    """Draw bounding boxes and labels on the frame"""
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
    
    # Count vehicles by type
    counts = {}
    class_column = 'detailed_class' if 'detailed_class' in detections.columns else 'name'
    for _, detection in detections.iterrows():
        class_name = detection[class_column]
        if class_name in counts:
            counts[class_name] += 1
        else:
            counts[class_name] = 1
    
    # Draw vehicle counts
    y_pos = 30
    for vehicle_type, count in counts.items():
        cv2.putText(frame, f"{vehicle_type}: {count}", (10, y_pos), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
    
    # Display FPS if enabled
    if show_fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("Vehicle Detection for Raspberry Pi")
    print("---------------------------------")
    print(f"Model: {args.model}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Detailed classification: {'Enabled' if args.detailed else 'Disabled'}")
    print(f"Headless mode: {'Enabled' if args.headless else 'Disabled'}")
    print("---------------------------------")
    
    # Download and load model
    print("Setting up vehicle detection model...")
    model_path = download_yolov5_model(args.model)
    model = load_model(model_path, conf=args.conf)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    print("Model loaded successfully!")
    
    # Initialize video stream thread
    print("Starting camera...")
    vs = VideoStreamThread(src=args.camera, width=args.width, height=args.height)
    if not vs.start():
        print("Failed to start video stream. Exiting.")
        return
    
    # Allow camera to warm up
    time.sleep(2.0)
    
    print("Processing video stream. Press 'q' to quit...")
    
    # Initialize variables
    frame_count = 0
    skip_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Get frame from video stream
            frame = vs.read()
            if frame is None:
                print("Error: Failed to get frame from camera.")
                break
                
            # Process every nth frame for better performance
            if skip_count < args.skip_frames:
                skip_count += 1
                
                # Still display the frame even if we skip detection
                if not args.headless:
                    cv2.imshow('Vehicle Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            else:
                skip_count = 0
            
            # Update fps calculation
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10 / max(1e-5, (end_time - start_time))
                start_time = end_time
                
            # Detect vehicles
            detections = detect_vehicles(model, frame, args.detailed)
            
            # Draw detections on the frame
            frame_with_detections = draw_detections(frame.copy(), detections, args.fps, fps)
            
            # Display the frame
            if not args.headless:
                cv2.imshow('Vehicle Detection', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print detection results to console in headless mode
            if args.headless and frame_count % 30 == 0:
                # Count vehicles by type
                counts = {}
                class_column = 'detailed_class' if 'detailed_class' in detections.columns else 'name'
                for _, detection in detections.iterrows():
                    class_name = detection[class_column]
                    if class_name in counts:
                        counts[class_name] += 1
                    else:
                        counts[class_name] = 1
                
                # Print counts
                print(f"FPS: {fps:.1f} | Detected vehicles: ", end="")
                for vehicle_type, count in counts.items():
                    print(f"{vehicle_type}: {count}, ", end="")
                print()
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        print("Cleaning up...")
        vs.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    # Import pandas here to avoid slow startup
    import pandas as pd
    main()