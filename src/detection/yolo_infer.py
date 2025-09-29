import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

class YOLOInference:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLO model for inference
        
        Args:
            model_path (str): Path to YOLO model weights
        """
        self.model = YOLO(model_path)
        
    def detect_frame(self, frame, conf_threshold=0.5, save_path=None):
        """
        Detect objects in a single frame
        
        Args:
            frame (numpy.ndarray): Input frame
            conf_threshold (float): Confidence threshold for detections
            save_path (str): Path to save annotated frame (optional)
            
        Returns:
            numpy.ndarray: Annotated frame with detections
        """
        # Run inference
        results = self.model(frame, conf=conf_threshold)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        
        # Save annotated frame if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_frame)
            print(f"Annotated frame saved to: {save_path}")
            
        return annotated_frame
    
    def extract_and_detect_frame(self, video_path, frame_number=0, 
                                conf_threshold=0.5, save_path=None):
        """
        Extract a specific frame from video and run detection
        
        Args:
            video_path (str): Path to video file
            frame_number (int): Frame number to extract (0-based)
            conf_threshold (float): Confidence threshold for detections
            save_path (str): Path to save annotated frame (optional)
            
        Returns:
            numpy.ndarray: Annotated frame with detections
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        
        # Validate frame number
        if frame_number >= total_frames or frame_number < 0:
            cap.release()
            raise ValueError(f"Invalid frame number. Must be between 0 and {total_frames-1}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the specific frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Error reading frame {frame_number}")
            
        print(f"Extracted frame {frame_number} from video")
        
        # Run detection on the frame
        annotated_frame = self.detect_frame(frame, conf_threshold, save_path)
        
        return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="YOLO inference on single video frame")
    parser.add_argument("--video", "-v", type=str, default="football_analytics/data/raw/Match_2031_5_0_test.mp4",
                       help="Path to video file")
    parser.add_argument("--frame", "-f", type=int, default=2000,
                       help="Frame number to extract (default: 0)")
    parser.add_argument("--model", "-m", type=str, default="football_analytics/models/yolo/best.pt",
                       help="Path to YOLO model (default: football_analytics/models/yolo/best.pt)")
    parser.add_argument("--conf", "-c", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument("--output", "-o", type=str,default='football_analytics/data/interim/annotated_frame.jpg',
                       help="Output path for annotated frame")
    parser.add_argument("--show", "-s", action="store_true",
                       help="Display the annotated frame")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        # Initialize YOLO inference
        yolo_detector = YOLOInference(args.model)
        
        # Extract frame and run detection
        annotated_frame = yolo_detector.extract_and_detect_frame(
            video_path=args.video,
            frame_number=args.frame,
            conf_threshold=args.conf,
            save_path=args.output
        )
        
        # Display frame if requested
        if args.show:
            cv2.imshow(f"YOLO Detection - Frame {args.frame}", annotated_frame)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        print("Detection completed successfully!")
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")

if __name__ == "__main__":
    main()