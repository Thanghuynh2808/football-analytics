import torch
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import cv2
from typing import List, Tuple, Optional
import pickle
import os

class FootballTeamClassifier:
    def __init__(self, device: str = "auto"):
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.is_fitted = False
        
        # Enhanced clustering models
        self.clustering_model = None
        self.scaler = None
        self.algorithm_type = None
        self.team_colors = None

    def extract_enhanced_color_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract enhanced color features with better jersey area detection and spatial information
        """
        if crop.size == 0:
            return np.zeros(15)  # Updated feature dimension
        
        try:
            # Resize and apply Gaussian blur to reduce noise
            crop_resized = cv2.resize(crop, (64, 128))
            crop_blurred = cv2.GaussianBlur(crop_resized, (3, 3), 0)
            
            # More precise jersey area detection (center region + upper body)
            h, w = crop_resized.shape[:2]
            jersey_region = crop_blurred[int(h*0.15):int(h*0.65), int(w*0.2):int(w*0.8)]
            
            # Convert to multiple color spaces for better representation
            hsv_crop = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
            lab_crop = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2LAB)
            
            # Mask out potential background using saturation threshold
            saturation = hsv_crop[:, :, 1]
            mask = saturation > np.percentile(saturation, 30)  # Keep high saturation areas
            
            if np.sum(mask) < 10:  # Fallback if mask too restrictive
                mask = np.ones_like(saturation, dtype=bool)
            
            # Extract dominant colors using K-means on masked pixels
            masked_pixels = hsv_crop[mask]
            if len(masked_pixels) > 10:
                kmeans_color = KMeans(n_clusters=2, random_state=42, n_init=5)
                kmeans_color.fit(masked_pixels)
                dominant_colors = kmeans_color.cluster_centers_
                
                # Color features: dominant H, S, V values
                color_features = dominant_colors.flatten()[:6]  # Top 2 dominant colors
            else:
                color_features = np.zeros(6)
            
            # Add color variance and spatial distribution features
            h_var = np.var(hsv_crop[:, :, 0][mask]) if np.sum(mask) > 0 else 0
            s_mean = np.mean(hsv_crop[:, :, 1][mask]) if np.sum(mask) > 0 else 0
            v_mean = np.mean(hsv_crop[:, :, 2][mask]) if np.sum(mask) > 0 else 0
            
            # LAB color space features for better color distinction
            l_mean = np.mean(lab_crop[:, :, 0][mask]) if np.sum(mask) > 0 else 0
            a_mean = np.mean(lab_crop[:, :, 1][mask]) if np.sum(mask) > 0 else 0
            b_mean = np.mean(lab_crop[:, :, 2][mask]) if np.sum(mask) > 0 else 0
            
            additional_features = np.array([h_var, s_mean, v_mean, l_mean, a_mean, b_mean])
            
            # Combine all features
            final_features = np.concatenate([color_features, additional_features])
            
            return final_features[:15]  # Ensure consistent size
            
        except Exception as e:
            print(f"Error extracting enhanced color features: {e}")
            return np.zeros(15)

    def collect_high_quality_training_data(self, video_path: str, model, stride: int = 30, max_crops: int = 500) -> List[np.ndarray]:
        """Collect high-quality player crops with better filtering"""
        print(f"Collecting high-quality training data from {video_path}")
        
        frame_generator = sv.get_video_frames_generator(source_path=video_path, stride=stride)
        crops = []
        frame_count = 0
        rejected_count = 0
        
        for frame in tqdm(frame_generator, desc="Collecting crops"):
            if len(crops) >= max_crops:
                break
                
            # Use higher confidence for training data quality
            result = model.predict(frame, conf=0.25, verbose=False)[0]
            
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            detections = detections.with_nms(threshold=0.3, class_agnostic=True)
            
            player_detections = detections[detections.class_id == 0]
            
            if len(player_detections) > 0:
                for i, xyxy in enumerate(player_detections.xyxy):
                    # Enhanced size and aspect ratio filtering
                    w = xyxy[2] - xyxy[0]
                    h = xyxy[3] - xyxy[1]
                    aspect_ratio = h / w if w > 0 else 0
                    conf = player_detections.confidence[i]
                    
                    # Better filtering criteria
                    if (w > 30 and h > 50 and  # Minimum size
                        aspect_ratio > 1.2 and aspect_ratio < 5.0 and  # Human-like aspect ratio
                        conf > 0.3 and  # High confidence only
                        w < frame.shape[1] * 0.4 and h < frame.shape[0] * 0.9):  # Not too large
                        
                        crop = sv.crop_image(frame, xyxy)
                        
                        # Additional quality checks
                        if self.is_good_quality_crop(crop):
                            crops.append(crop)
                        else:
                            rejected_count+=1
                    else:
                        rejected_count+=1

            frame_count += 1
        
        # Remove similar crops to increase diversity
        crops = self.remove_similar_crops(crops,similarity_threshold=0.9)
        
        print(f"Collected {len(crops)} high-quality player crops from {frame_count} frames")
        return crops

    def is_good_quality_crop(self, crop: np.ndarray) -> bool:
        """Check if crop meets quality standards"""
        if crop.size == 0:
            return False
        
        # Check for blur (Laplacian variance)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check color diversity (avoid mostly black/white crops)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation_mean = np.mean(hsv[:, :, 1])
        
        return blur_score > 50 and saturation_mean > 20

    def remove_similar_crops(self, crops: List[np.ndarray], similarity_threshold: float = 0.8) -> List[np.ndarray]:
        """Remove visually similar crops to increase training diversity"""
        if len(crops) <= 1:
            return crops
        
        # Simple approach: compute histograms and remove similar ones
        filtered_crops = [crops[0]]
        
        for crop in crops[1:]:
            is_different = True
            crop_hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            
            for existing_crop in filtered_crops[-10:]:  # Check against recent crops only
                existing_hist = cv2.calcHist([existing_crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                correlation = cv2.compareHist(crop_hist, existing_hist, cv2.HISTCMP_CORREL)
                
                if correlation > similarity_threshold:
                    is_different = False
                    break
            
            if is_different:
                filtered_crops.append(crop)
        
        return filtered_crops

    def fit_with_validation(self, video_path: str, model, **kwargs):
        """Enhanced training with validation and multiple clustering approaches"""
        crops = self.collect_high_quality_training_data(video_path, model, **kwargs)
        
        if len(crops) < 20:
            raise ValueError(f"Not enough training data. Only collected {len(crops)} crops")
        
        print(f"Extracting enhanced features from {len(crops)} crops...")
        
        # Extract enhanced features
        features = []
        for crop in tqdm(crops, desc="Extracting features"):
            feature = self.extract_enhanced_color_features(crop)
            features.append(feature)
        
        features = np.array(features)
        
        # Feature scaling for better clustering
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        print(f"Feature shape: {features_scaled.shape}")
        
        # Try multiple clustering algorithms
        algorithms = {
            'kmeans': KMeans(n_clusters=2, random_state=42, n_init=10),
            'gmm': GaussianMixture(n_components=2, random_state=42)
        }
        
        best_score = -1
        best_model = None
        best_algorithm = None
        
        for name, algorithm in algorithms.items():
            try:
                if name == 'gmm':
                    labels = algorithm.fit_predict(features_scaled)
                else:
                    labels = algorithm.fit_predict(features_scaled)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features_scaled, labels)
                    
                    # Additional validation: check cluster balance
                    cluster_counts = np.bincount(labels)
                    balance_ratio = min(cluster_counts) / max(cluster_counts)
                    
                    # Penalize heavily imbalanced clusters
                    if balance_ratio < 0.05:
                        score *= 0.7
                    
                    print(f"Algorithm: {name}, Silhouette: {score:.3f}, Balance: {balance_ratio:.3f}")
                    print(f"Team distribution: {cluster_counts}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = algorithm
                        best_algorithm = name
                        
            except Exception as e:
                print(f"Failed clustering with {name}: {e}")
                continue
        
        if best_model is None:
            # Fallback
            best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
            best_model.fit(features_scaled)
            best_algorithm = 'kmeans_fallback'
            print("Using fallback 2-cluster KMeans solution")
        
        self.clustering_model = best_model
        self.algorithm_type = best_algorithm
        labels = best_model.fit_predict(features_scaled) if best_algorithm == 'kmeans_fallback' else best_model.predict(features_scaled)
        
        print(f"Best algorithm: {best_algorithm} with silhouette score: {best_score:.3f}")
        print(f"Final team distribution: {np.bincount(labels)}")
        
        self.is_fitted = True
        return True

    # Legacy methods for backward compatibility
    def collect_training_data(self, video_path: str, model, **kwargs):
        """Legacy method - calls new high-quality data collection"""
        return self.collect_high_quality_training_data(video_path, model, **kwargs)

    def extract_color_feature(self, crop: np.ndarray) -> np.ndarray:
        """Legacy method - calls enhanced feature extraction"""
        return self.extract_enhanced_color_features(crop)

    def fit(self, video_path: str, model, **kwargs):
        """Main training method using enhanced approach"""
        return self.fit_with_validation(video_path, model, **kwargs)
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Enhanced prediction with feature scaling"""
        if not self.is_fitted:
            raise ValueError("Classifier not fitted yet!")
        
        if len(crops) == 0:
            return np.array([])
        
        # Extract enhanced features
        features = []
        for crop in crops:
            feature = self.extract_enhanced_color_features(crop)
            features.append(feature)
        
        features = np.array(features)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.clustering_model.predict(features_scaled)
        return predictions
    
    def save_classifier(self, path: str):
        """Save trained classifier with all components"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'is_fitted': self.is_fitted,
            'clustering_model': self.clustering_model,
            'scaler': self.scaler,
            'algorithm_type': self.algorithm_type,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Enhanced classifier saved to {path}")
    
    def load_classifier(self, path: str):
        """Load trained classifier with all components"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier file {path} not found")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.is_fitted = save_data['is_fitted']
        self.clustering_model = save_data.get('clustering_model', save_data.get('kmeans_model'))  # Backward compatibility
        self.scaler = save_data.get('scaler')
        self.algorithm_type = save_data.get('algorithm_type', 'kmeans')
        
        # Handle old format
        if self.scaler is None:
            print("Warning: Loading old classifier format. Feature scaling not available.")
            self.scaler = StandardScaler()  # Create dummy scaler
        
        print(f"Enhanced classifier loaded from {path}")

    def get_cluster_info(self) -> dict:
        """Get information about the trained clusters"""
        if not self.is_fitted:
            return {}
        
        info = {
            'algorithm': self.algorithm_type,
            'n_clusters': 2,
            'is_fitted': self.is_fitted
        }
        
        if hasattr(self.clustering_model, 'cluster_centers_'):
            info['cluster_centers'] = self.clustering_model.cluster_centers_
        
        return info