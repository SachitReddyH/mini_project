import mediapipe as mp
import numpy as np
import cv2

class HandTracker:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand tracker with MediaPipe.
        
        Args:
            static_mode: If True, detection is performed on every frame
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum confidence for hand detection
            tracking_confidence: Minimum confidence for hand tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image.
        
        Args:
            img: Input image
            draw: If True, draw landmarks on the image
            
        Returns:
            img: Image with landmarks drawn (if draw=True)
            hand_landmarks: List of hand landmarks
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Initialize empty list for hand landmarks
        hand_landmarks = []
        
        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmark in enumerate(self.results.multi_hand_landmarks):
                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmark, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Extract landmarks
                landmarks = []
                for lm in hand_landmark.landmark:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy, lm.z])
                
                hand_landmarks.append(landmarks)
        
        return img, hand_landmarks
    
    def get_hand_angles(self, hand_landmarks):
        """
        Calculate angles between fingers for gesture recognition.
        
        Args:
            hand_landmarks: List of hand landmarks
            
        Returns:
            angles: Dictionary of angles between fingers
        """
        if not hand_landmarks or len(hand_landmarks) == 0:
            return None
        
        # Get the first hand's landmarks
        landmarks = hand_landmarks[0]
        
        # Define finger indices
        thumb_tip = 4
        index_tip = 8
        middle_tip = 12
        ring_tip = 16
        pinky_tip = 20
        wrist = 0
        
        # Calculate vectors
        thumb_vector = np.array(landmarks[thumb_tip][:2]) - np.array(landmarks[wrist][:2])
        index_vector = np.array(landmarks[index_tip][:2]) - np.array(landmarks[wrist][:2])
        middle_vector = np.array(landmarks[middle_tip][:2]) - np.array(landmarks[wrist][:2])
        ring_vector = np.array(landmarks[ring_tip][:2]) - np.array(landmarks[wrist][:2])
        pinky_vector = np.array(landmarks[pinky_tip][:2]) - np.array(landmarks[wrist][:2])
        
        # Calculate angles
        angles = {
            'thumb_index': self._angle_between_vectors(thumb_vector, index_vector),
            'index_middle': self._angle_between_vectors(index_vector, middle_vector),
            'middle_ring': self._angle_between_vectors(middle_vector, ring_vector),
            'ring_pinky': self._angle_between_vectors(ring_vector, pinky_vector)
        }
        
        return angles
    
    def _angle_between_vectors(self, v1, v2):
        """Calculate the angle between two vectors in degrees."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0
        
        cos_angle = dot_product / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def recognize_sign(self, hand_landmarks):
        """
        Recognize a sign based on hand landmarks.
        
        Args:
            hand_landmarks: List of hand landmarks
            
        Returns:
            sign: Recognized sign
            confidence: Confidence score
        """
        if not hand_landmarks or len(hand_landmarks) == 0:
            return "No hand detected", 0.0
        
        # Get hand angles
        angles = self.get_hand_angles(hand_landmarks)
        
        if angles is None:
            return "No hand detected", 0.0
        
        # Simple rule-based recognition (can be replaced with a trained model)
        # This is a placeholder for demonstration purposes
        if angles['thumb_index'] < 30 and angles['index_middle'] > 150:
            return "A", 0.9
        elif angles['thumb_index'] > 150 and angles['index_middle'] < 30:
            return "B", 0.9
        elif angles['thumb_index'] < 30 and angles['index_middle'] < 30:
            return "C", 0.9
        else:
            return "Unknown", 0.5 