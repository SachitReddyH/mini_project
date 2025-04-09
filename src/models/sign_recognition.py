import numpy as np
import cv2
import tensorflow as tf
from .hand_tracking import HandTracker

class SignLanguageRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the sign language recognizer.
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.hand_tracker = HandTracker()
        self.model = None
        
        # Load pre-trained model if provided
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using rule-based recognition instead")
    
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
    
    def recognize_sign(self, image):
        """
        Recognize a sign from an image.
        
        Args:
            image: Input image
            
        Returns:
            sign: Recognized sign
            confidence: Confidence score
        """
        # If we have a trained model, use it
        if self.model is not None:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(np.expand_dims(processed_image, axis=0))
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Map class index to sign (this would be based on your training data)
            sign_mapping = {
                0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
                5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
                10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
                15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
                20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
                25: "Z"
            }
            
            sign = sign_mapping.get(predicted_class, "Unknown")
            
            return sign, float(confidence)
        
        # Otherwise, use the rule-based recognition from the hand tracker
        _, hand_landmarks = self.hand_tracker.find_hands(image, draw=False)
        return self.hand_tracker.recognize_sign(hand_landmarks)
    
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