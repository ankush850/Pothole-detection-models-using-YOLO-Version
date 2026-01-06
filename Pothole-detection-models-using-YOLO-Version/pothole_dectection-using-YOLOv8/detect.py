

from ultralytics import YOLO
import cv2
import argparse
import os
from pathlib import Path
import time

class PotholeDetector:
    def __init__(self, model_path='best.pt', conf_thresh=0.25):
        """
        Initialize pothole detector
        
        Args:
            model_path: Path to trained model
            conf_thresh: Confidence threshold
        """
        print(f"📦 Loading model: {model_path}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try to find best model in runs
            runs_path = Path('runs/detect')
            if runs_path.exists():
                model_dirs = list(runs_path.glob('pothole_v8*'))
                if model_dirs:
                    latest_model = max(model_dirs, key=os.path.getmtime)
                    model_path = latest_model / 'weights' / 'best.pt'
                    if model_path.exists():
                        print(f"Found model: {model_path}")
        
        if not os.path.exists(model_path):
            print("❌ Model not found! Train a model first with train.py")
            print("Or specify path with --model")
            self.model = None
            return
        
        # Load model
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.class_names = ['pothole']
        
        print("✅ Model loaded successfully!")
    
    def detect_image(self, image_path, output_dir='results'):
        """
        Detect potholes in a single image
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
        """
        if self.model is None:
            return
        
        print(f"🔍 Processing: {image_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.conf_thresh,
            iou=0.45,
            imgsz=640,
            save=True,
            save_dir=output_dir,
            save_txt=True,
            save_conf=True
        )
        
        # Count detections
        detections = 0
        for result in results:
            if result.boxes is not None:
                detections = len(result.boxes)
        
        print(f"✅ Found {detections} pothole(s)")
        
        # Show image if in interactive mode
        for result in results:
            img = result.plot()
            cv2.imshow('Pothole Detection', img)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        
        return results
    
    def detect_video(self, video_path, output_path='results/output.mp4'):
        """
        Detect potholes in video
        
        Args:
            video_path: Path to input video
            output_path: Output video path
        """
        if self.model is None:
            return
        
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing frames... (Press 'q' to stop)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_thresh,
                imgsz=640,
                verbose=False
            )
            
            # Draw results
            for result in results:
                annotated_frame = result.plot()
                out.write(annotated_frame)
                
                # Display
                cv2.imshow('Pothole Detection', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count} frames ({fps_actual:.1f} FPS)")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Video processing complete!")
        print(f"📊 Total frames: {frame_count}")
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📁 Output saved: {output_path}")
    
    def detect_realtime(self, camera_id=0):
        """
        Real-time pothole detection from webcam
        
        Args:
            camera_id: Camera device ID
        """
        if self.model is None:
            return
        
        print("📹 Starting real-time detection...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Variables for FPS calculation
        prev_time = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            current_time = time.time()
            if prev_time:
                fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Run inference
            results = self.model(
                frame,
                conf=self.conf_thresh,
                imgsz=640,
                verbose=False
            )
            
            # Draw results
            for result in results:
                annotated_frame = result.plot()
                
                # Display FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display detections count
                if result.boxes is not None:
                    count = len(result.boxes)
                    cv2.putText(
                        annotated_frame,
                        f"Potholes: {count}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                cv2.imshow('Real-time Pothole Detection', annotated_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Real-time detection stopped")

def main():
    parser = argparse.ArgumentParser(description='Pothole Detection')
    parser.add_argument('--source', type=str, required=True,
                       help='Image/Video path or camera ID (0 for webcam)')
    parser.add_argument('--model', type=str, default='runs/detect/pothole_v8m/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🕳️  Pothole Detection")
    print("="*60)
    
    # Create detector
    detector = PotholeDetector(model_path=args.model, conf_thresh=args.conf)
    
    if detector.model is None:
        return
    
    # Determine source type
    source = args.source
    
    if source.isdigit():
        # Webcam
        detector.detect_realtime(camera_id=int(source))
    
    elif source.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Image
        detector.detect_image(source, args.output)
    
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video
        output_path = os.path.join(args.output, 'output.mp4')
        detector.detect_video(source, output_path)
    
    else:
        print(f"❌ Unknown source type: {source}")
        print("Supported: image files, video files, or camera ID")

if __name__ == "__main__":
    main()