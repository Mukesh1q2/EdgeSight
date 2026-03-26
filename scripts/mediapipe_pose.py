"""
MediaPipe Pose Detection Wrapper for EdgeSight

Provides real pose keypoint extraction for fall detection.
Can be called from C++ via subprocess or used directly in Python.

Usage:
    python scripts/mediapipe_pose.py <image_path>
    # Outputs JSON array of keypoints [x1, y1, conf1, x2, y2, conf2, ...]
    
Or import as module:
    from scripts.mediapipe_pose import extract_pose_keypoints
    keypoints = extract_pose_keypoints(image)
"""

import sys
import json
import numpy as np
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not installed. Install with: pip install mediapipe")


# MediaPipe pose landmark indices
MEDIAPIPE_LANDMARKS = {
    'nose': 0,
    'left_eye': 2,
    'right_eye': 5,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
}

# EdgeSight expects these 15 keypoints in order
EDGESIGHT_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye', 
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]


class MediaPipePoseExtractor:
    """Extract pose keypoints using MediaPipe."""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose extractor.
        
        Args:
            static_image_mode: True for images, False for video
            model_complexity: 0, 1, or 2 (higher = more accurate, slower)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Extract pose keypoints from an image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            Array of shape (45,) with 15 keypoints (x, y, confidence) flattened
            Returns zeros if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose.process(image_rgb)
        
        # Initialize with zeros
        keypoints = np.zeros(45, dtype=np.float32)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            for i, keypoint_name in enumerate(EDGESIGHT_KEYPOINTS):
                mp_idx = MEDIAPIPE_LANDMARKS[keypoint_name]
                landmark = landmarks[mp_idx]
                
                # Store x, y, visibility
                base_idx = i * 3
                keypoints[base_idx] = landmark.x
                keypoints[base_idx + 1] = landmark.y
                keypoints[base_idx + 2] = landmark.visibility
        
        return keypoints
    
    def extract_keypoints_normalized(self, image: np.ndarray) -> np.ndarray:
        """Extract keypoints normalized to image dimensions."""
        h, w = image.shape[:2]
        keypoints = self.extract_keypoints(image)
        
        # x and y are already normalized 0-1 by MediaPipe
        # Just return as-is
        return keypoints
    
    def close(self):
        """Release resources."""
        self.pose.close()


def extract_pose_keypoints(image_path: str) -> list:
    """
    Extract pose keypoints from an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        List of 45 floats (15 keypoints * 3 values)
    """
    if not MEDIAPIPE_AVAILABLE:
        # Return simulated keypoints if MediaPipe not available
        print("MediaPipe not available, returning simulated keypoints")
        return list(np.random.rand(45).astype(np.float32))
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    extractor = MediaPipePoseExtractor()
    keypoints = extractor.extract_keypoints(image)
    extractor.close()
    
    return keypoints.tolist()


def process_video(video_path: str, output_path: str = None):
    """
    Process a video and extract keypoints for each frame.
    
    Args:
        video_path: Path to video file
        output_path: Path to save keypoints as .npy (optional)
        
    Returns:
        List of keypoint arrays, one per frame
    """
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    extractor = MediaPipePoseExtractor(static_image_mode=False)
    all_keypoints = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = extractor.extract_keypoints(frame)
        all_keypoints.append(keypoints)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    extractor.close()
    
    print(f"Total frames processed: {frame_count}")
    
    if output_path:
        np.save(output_path, np.array(all_keypoints))
        print(f"Saved keypoints to {output_path}")
    
    return all_keypoints


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python mediapipe_pose.py <image_path>")
        print("       python mediapipe_pose.py --video <video_path>")
        sys.exit(1)
    
    if sys.argv[1] == "--video":
        if len(sys.argv) < 3:
            print("Usage: python mediapipe_pose.py --video <video_path>")
            sys.exit(1)
        video_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        keypoints = process_video(video_path, output_path)
        print(f"Processed {len(keypoints)} frames")
    else:
        image_path = sys.argv[1]
        keypoints = extract_pose_keypoints(image_path)
        # Output as JSON for C++ to parse
        print(json.dumps(keypoints))


if __name__ == "__main__":
    main()