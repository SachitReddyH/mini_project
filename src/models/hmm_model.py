import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SignLanguageHMM:
    def __init__(self, n_states=5, n_iter=100):
        """
        Initialize the HMM model for sign language recognition.
        
        Args:
            n_states (int): Number of hidden states in the HMM
            n_iter (int): Maximum number of iterations for training
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            n_iter=n_iter,
            covariance_type="diag",
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def preprocess_features(self, features):
        """
        Preprocess the input features.
        
        Args:
            features (np.array): Input features array
            
        Returns:
            np.array: Preprocessed features
        """
        if not self.is_fitted:
            self.scaler.fit(features)
            self.is_fitted = True
        return self.scaler.transform(features)
    
    def train(self, features, lengths):
        """
        Train the HMM model on the provided features.
        
        Args:
            features (np.array): Training features
            lengths (list): Length of each sequence in features
        """
        preprocessed_features = self.preprocess_features(features)
        self.model.fit(preprocessed_features, lengths=lengths)
        
    def predict(self, features):
        """
        Predict the most likely sequence of states for the input features.
        
        Args:
            features (np.array): Input features
            
        Returns:
            tuple: (predicted_states, log_probability)
        """
        preprocessed_features = self.preprocess_features(features)
        return self.model.predict(preprocessed_features)
    
    def score(self, features):
        """
        Calculate the log probability of the features under the model.
        
        Args:
            features (np.array): Input features
            
        Returns:
            float: Log probability score
        """
        preprocessed_features = self.preprocess_features(features)
        return self.model.score(preprocessed_features)
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, model_path)
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            SignLanguageHMM: Loaded model instance
        """
        model_data = joblib.load(model_path)
        instance = cls(n_states=model_data['model'].n_components)
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = model_data['is_fitted']
        return instance

def extract_hand_features(hand_landmarks):
    """
    Extract features from hand landmarks for HMM input.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        np.array: Extracted features
    """
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features)

def prepare_sequence_data(hand_sequences):
    """
    Prepare sequence data for HMM training.
    
    Args:
        hand_sequences (list): List of hand landmark sequences
        
    Returns:
        tuple: (features, lengths)
    """
    features = []
    lengths = []
    
    for sequence in hand_sequences:
        sequence_features = []
        for landmarks in sequence:
            features = extract_hand_features(landmarks)
            sequence_features.append(features)
        features.extend(sequence_features)
        lengths.append(len(sequence))
    
    return np.array(features), lengths 