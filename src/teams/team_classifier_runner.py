#!/usr/bin/env python3
"""
Team Classifier Runner Script
This script demonstrates how to use the FootballTeamClassifier to:
1. Train a team classifier using video data
2. Apply the classifier to new frames
3. Visualize the results
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from team_classifier import FootballTeamClassifier
from ultralytics import YOLO
import supervision as sv

class TeamClassifierRunner:
    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        """
        Initialize the team classifier runner
        
        Args:
            yolo_model_path (str): Path to YOLO model weights
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.team_classifier = FootballTeamClassifier()
        
        # Team colors for visualization (BGR format)
        self.team_colors = [
            (0, 255, 0),    # Green for Team 0
            (0, 0, 255),    # Red for Team 1
            (255, 0, 0),    # Blue for additional teams
            (0, 255, 255),  # Yellow
        ]

    def train_classifier(self, video_path: str, save_path: str = None) -> bool:
        """
        Train the team classifier using a video
        
        Args:
            video_path (str): Path to training video
            save_path (str): Path to save trained classifier
            
        Returns:
            bool: True if training successful
        """
        print(f"Training team classifier using video: {video_path}")
        
        try:
            # Train the classifier
            success = self.team_classifier.fit(video_path, self.yolo_model)
            
            if success:
                print("✅ Team classifier training completed successfully!")
                
                # Save the classifier if path provided
                if save_path:
                    self.team_classifier.save_classifier(save_path)
                    print(f"✅ Classifier saved to: {save_path}")
                
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            return False

    def load_classifier(self, classifier_path: str) -> bool:
        """
        Load a pre-trained classifier
        
        Args:
            classifier_path (str): Path to saved classifier
            
        Returns:
            bool: True if loading successful
        """
        try:
            self.team_classifier.load_classifier(classifier_path)
            print(f"✅ Classifier loaded from: {classifier_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading classifier: {str(e)}")
            return False

    def extract_player_crops(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[List[int]]]:
        """
        Extract player crops from frame using YOLO
        
        Args:
            frame (np.ndarray): Input frame
            conf_threshold (float): Detection confidence threshold
            
        Returns:
            tuple: (crops, bboxes)
        """
        # Run YOLO detection
        results = self.yolo_model.predict(frame, conf=conf_threshold,iou=0.3, verbose=False)

        crops = []
        bboxes = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy().astype(int)
                
                # Filter for person/player detections (class 0)
                for i, class_id in enumerate(cls):
                    if class_id == 0 and conf[i] >= conf_threshold:
                        bbox = xyxy[i]
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Ensure coordinates are within bounds
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Filter by size
                        if (x2 - x1) > 30 and (y2 - y1) > 60:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                crops.append(crop)
                                bboxes.append([x1, y1, x2, y2])
        
        return crops, bboxes

    def classify_frame(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Tuple[np.ndarray, List[int], List[List[int]]]:
        """
        Classify players in a frame and return annotated result
        
        Args:
            frame (np.ndarray): Input frame
            conf_threshold (float): Detection confidence threshold
            
        Returns:
            tuple: (annotated_frame, team_predictions, bboxes)
        """
        if not self.team_classifier.is_fitted:
            raise ValueError("Team classifier not trained yet! Please train or load a classifier first.")
        
        # Extract player crops
        crops, bboxes = self.extract_player_crops(frame, conf_threshold)
        
        if len(crops) == 0:
            print("No players detected in frame")
            return frame.copy(), [], []
        
        # Predict teams
        team_predictions = self.team_classifier.predict(crops).tolist()
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw bounding boxes with team colors
        for i, (bbox, team_id) in enumerate(zip(bboxes, team_predictions)):
            x1, y1, x2, y2 = bbox
            color = self.team_colors[team_id % len(self.team_colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw team label
            label = f"Team {team_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), 
                         color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add summary
        unique_teams, counts = np.unique(team_predictions, return_counts=True)
        summary = f"Players: {len(crops)}, Teams: {len(unique_teams)}"
        cv2.putText(annotated_frame, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return annotated_frame, team_predictions, bboxes

    def analyze_video_frame(self, video_path: str, frame_number: int = 0, 
                          save_path: str = None, display: bool = True) -> bool:
        """
        Analyze a specific frame from video
        
        Args:
            video_path (str): Path to video file
            frame_number (int): Frame number to analyze
            save_path (str): Path to save result
            display (bool): Whether to display result
            
        Returns:
            bool: True if successful
        """
        try:
            # Read specific frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return False
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"❌ Cannot read frame {frame_number}")
                return False
            
            print(f"Analyzing frame {frame_number}...")
            
            # Classify frame
            annotated_frame, team_predictions, bboxes = self.classify_frame(frame)
            
            # Print results
            if len(team_predictions) > 0:
                unique_teams, counts = np.unique(team_predictions, return_counts=True)
                print(f"✅ Found {len(team_predictions)} players:")
                for team, count in zip(unique_teams, counts):
                    print(f"   Team {team}: {count} players")
            else:
                print("No players detected in frame")
            
            # Save result
            if save_path:
                cv2.imwrite(save_path, annotated_frame)
                print(f"✅ Result saved to: {save_path}")
            
            # Display result
            if display:
                cv2.imshow(f"Team Classification - Frame {frame_number}", annotated_frame)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing frame: {str(e)}")
            return False

    def create_team_samples_visualization(self, video_path: str, num_frames: int = 10) -> bool:
        """
        Create a visualization showing sample players from each team
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to sample
            
        Returns:
            bool: True if successful
        """
        if not self.team_classifier.is_fitted:
            print("❌ Team classifier not trained yet!")
            return False
        
        try:
            print(f"Creating team samples from {num_frames} frames...")
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            team_samples = {0: [], 1: []}  # Assuming 2 teams
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    crops, _ = self.extract_player_crops(frame)
                    if len(crops) > 0:
                        predictions = self.team_classifier.predict(crops)
                        
                        for crop, team_id in zip(crops, predictions):
                            if len(team_samples[team_id]) < 10:  # Limit samples per team
                                # Resize crop for visualization
                                resized_crop = cv2.resize(crop, (80, 120))
                                team_samples[team_id].append(resized_crop)
            
            cap.release()
            
            # Create visualization
            fig, axes = plt.subplots(2, 10, figsize=(20, 6))
            fig.suptitle('Team Classification Samples', fontsize=16)
            
            for team_id in range(2):
                samples = team_samples[team_id]
                for i in range(10):
                    if i < len(samples):
                        # Convert BGR to RGB for matplotlib
                        rgb_image = cv2.cvtColor(samples[i], cv2.COLOR_BGR2RGB)
                        axes[team_id, i].imshow(rgb_image)
                    else:
                        axes[team_id, i].set_facecolor('black')
                    
                    axes[team_id, i].set_xticks([])
                    axes[team_id, i].set_yticks([])
                    
                    if i == 0:
                        axes[team_id, i].set_ylabel(f'Team {team_id}', fontsize=12)
            
            # Save visualization
            save_path = "football_analytics/data/interim/team_samples_visualization.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.show()
            
            print(f"✅ Team samples visualization saved to: {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Football Team Classifier Runner")
    parser.add_argument("--video", "-v", type=str, 
                       default="D:\\Programing\\football_analytics\\data\\raw\\Match_2031_5_0_test.mp4",
                       help="Path to video file")
    parser.add_argument("--model", "-m", type=str, 
                       default="D:\\Programing\\football_analytics\\models\\yolo\\best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--frame", "-f", type=int, default=3000,
                       help="Frame number to analyze (default: 3000)")
    parser.add_argument("--train", "-t", action="store_true",
                       help="Train new team classifier")
    parser.add_argument("--classifier", "-c", type=str,
                       default="D:\\Programing\\football_analytics\\models\\classifier\\team_classifier.pkl",
                       help="Path to save/load classifier")
    parser.add_argument("--output", "-o", type=str,
                       default="D:\\Programing\\football_analytics\\data\\interim\\team_classified_frame.jpg",
                       help="Output path for result")
    parser.add_argument("--show", "-s", action="store_true",
                       help="Display results")
    parser.add_argument("--samples", action="store_true",
                       help="Create team samples visualization")
    
    args = parser.parse_args()
    
    print("🏈 Football Team Classifier Runner")
    print("=" * 50)
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    try:
        # Initialize runner
        runner = TeamClassifierRunner(args.model)
        
        # Train or load classifier
        if args.train:
            print("🎯 Training new team classifier...")
            success = runner.train_classifier(args.video, args.classifier)
            if not success:
                print("❌ Training failed!")
                return
        else:
            # Try to load existing classifier
            if os.path.exists(args.classifier):
                print("📂 Loading existing classifier...")
                success = runner.load_classifier(args.classifier)
                if not success:
                    print("⚠️  Loading failed, training new classifier...")
                    success = runner.train_classifier(args.video, args.classifier)
                    if not success:
                        return
            else:
                print("🎯 No existing classifier found, training new one...")
                success = runner.train_classifier(args.video, args.classifier)
                if not success:
                    return
        
        # Analyze frame
        print(f"\n🔍 Analyzing frame {args.frame}...")
        runner.analyze_video_frame(args.video, args.frame, args.output, args.show)
        
        # Create samples visualization if requested
        if args.samples:
            print("\n🎨 Creating team samples visualization...")
            runner.create_team_samples_visualization(args.video)
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
