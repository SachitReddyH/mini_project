import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import os

class DataPreprocessor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_video(self, video_path, output_dir, frame_interval=5):
        """
        Process a video file to extract hand gestures
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Process frame
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    # Save processed frame
                    output_file = output_path / f"frame_{frame_count:04d}.jpg"
                    cv2.imwrite(str(output_file), processed_frame)

            frame_count += 1

        cap.release()
        return frame_count

    def process_frame(self, frame):
        """
        Process a single frame to detect and extract hand landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Create a blank image for drawing
            output_frame = np.zeros_like(frame)
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    output_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract hand region
                h, w, _ = frame.shape
                landmarks = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                
                # Add padding
                padding = 20
                x_min = max(0, int(x_min) - padding)
                y_min = max(0, int(y_min) - padding)
                x_max = min(w, int(x_max) + padding)
                y_max = min(h, int(y_max) + padding)
                
                # Extract hand region
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                # Resize to standard size
                hand_region = cv2.resize(hand_region, (64, 64))
                
                return hand_region
        
        return None

    def create_dataset(self, data_dir, output_dir):
        """
        Create a dataset from processed frames
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process all videos in the data directory
        for video_file in data_path.glob("*.mp4"):
            print(f"Processing {video_file}")
            video_output_dir = output_path / video_file.stem
            self.process_video(str(video_file), str(video_output_dir))

    def augment_data(self, image):
        """
        Apply data augmentation to increase dataset size
        """
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))
        
        # Rotation
        for angle in [15, -15]:
            matrix = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (64, 64))
            augmented_images.append(rotated)
        
        # Brightness adjustment
        for alpha in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            augmented_images.append(adjusted)
        
        return augmented_images

    def prepare_training_data(self, data_dir, output_dir):
        """
        Prepare the final training dataset
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process all processed frames
        for frame_dir in data_path.glob("*"):
            if frame_dir.is_dir():
                for frame_file in frame_dir.glob("*.jpg"):
                    # Read frame
                    frame = cv2.imread(str(frame_file))
                    
                    # Apply augmentation
                    augmented_frames = self.augment_data(frame)
                    
                    # Save augmented frames
                    for i, aug_frame in enumerate(augmented_frames):
                        output_file = output_path / f"{frame_file.stem}_aug_{i}.jpg"
                        cv2.imwrite(str(output_file), aug_frame)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Process videos
    preprocessor.create_dataset("data/raw", "data/processed")
    
    # Prepare training data
    preprocessor.prepare_training_data("data/processed", "data/processed/augmented")