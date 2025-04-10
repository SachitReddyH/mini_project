import numpy as np
import cv2
import tensorflow as tf
from .hand_tracking import HandTracker
from .hmm_model import SignLanguageHMM, prepare_sequence_data

class SignLanguageRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the sign language recognizer.
        
        Args:
            model_path (str, optional): Path to pre-trained HMM model
        """
        self.hand_tracker = HandTracker()
        self.hmm_model = SignLanguageHMM()
        if model_path:
            self.hmm_model = SignLanguageHMM.load_model(model_path)
        
        # Dictionary mapping sign indices to labels
        self.sign_labels = {
            0: "Hello",
            1: "Thank You",
            2: "Please",
            3: "Yes",
            4: "No",
            # Add more signs as needed
        }
        
    def preprocess_frame(self, frame):
        """
        Preprocess the input frame.
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            np.array: Preprocessed frame
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        frame_rgb = self.hand_tracker.find_hands(frame_rgb)
        
        return frame_rgb
    
    def recognize_sign(self, frame):
        """
        Recognize sign language from a frame.
        
        Args:
            frame (np.array): Input frame
            
        Returns:
            tuple: (predicted_sign, confidence)
        """
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Get hand landmarks
        hand_landmarks = self.hand_tracker.find_positions(processed_frame)
        
        if not hand_landmarks:
            return "No hand detected", 0.0
        
        # Extract features and prepare for HMM
        features, _ = prepare_sequence_data([hand_landmarks])
        
        # Get prediction from HMM
        predicted_states = self.hmm_model.predict(features)
        
        # Get the most common state as the predicted sign
        predicted_sign_idx = np.bincount(predicted_states).argmax()
        
        # Calculate confidence score
        confidence = self.hmm_model.score(features)
        confidence = 1 / (1 + np.exp(-confidence))  # Convert to probability
        
        return self.sign_labels.get(predicted_sign_idx, "Unknown"), confidence
    
    def train(self, training_data):
        """
        Train the HMM model on new data.
        
        Args:
            training_data (list): List of hand landmark sequences
        """
        features, lengths = prepare_sequence_data(training_data)
        self.hmm_model.train(features, lengths)
    
    def save_model(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        self.hmm_model.save_model(model_path)
    
    def preprocess_image(self, image):
        """
        Preprocess the image for recognition.
        
        Args:
            image: Input image
            
        Returns:
            processed_image: Preprocessed image
        """
        # Resize image to a standard size
        processed_image = cv2.resize(image, (224, 224))
        
        # Convert to RGB
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        processed_image = processed_image / 255.0
        
        return processed_image
    
    def extract_features(self, image):
        """
        Extract features from the image using hand tracking.
        
        Args:
            image: Input image
            
        Returns:
            features: Extracted features
        """
        # Find hands in the image
        _, hand_landmarks = self.hand_tracker.find_hands(image, draw=False)
        
        # If no hands detected, return None
        if not hand_landmarks or len(hand_landmarks) == 0:
            return None
        
        # Get hand angles
        angles = self.hand_tracker.get_hand_angles(hand_landmarks)
        
        if angles is None:
            return None
        
        # Convert angles to a feature vector
        features = np.array([
            angles['thumb_index'],
            angles['index_middle'],
            angles['middle_ring'],
            angles['ring_pinky']
        ])
        
        return features
    
    def train_model(self, dataset_path, epochs=10, batch_size=32):
        """
        Train a model on a dataset of sign language images.
        
        Args:
            dataset_path: Path to the dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            model: Trained model
        """
        # This is a placeholder for the actual training code
        # In a real implementation, you would:
        # 1. Load and preprocess the dataset
        # 2. Define a model architecture
        # 3. Train the model
        # 4. Save the model
        
        print("Training a model would be implemented here")
        print(f"Dataset path: {dataset_path}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        # For now, we'll just create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')  # 26 letters in the alphabet
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created (but not trained)")
        
        return model 