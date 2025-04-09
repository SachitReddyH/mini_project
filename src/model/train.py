import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..data_processing.preprocess import DataPreprocessor
from .sign_language_model import SignLanguageTranslator, TextToSignGenerator

def load_data(data_dir):
    """
    Load and prepare the dataset
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    
    # Load images and labels
    for label_dir in data_path.glob("*"):
        if label_dir.is_dir():
            label = int(label_dir.name)
            for image_file in label_dir.glob("*.jpg"):
                image = tf.keras.preprocessing.image.load_img(
                    str(image_file),
                    target_size=(64, 64)
                )
                image = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def train_sign_language_model(data_dir, model_dir, batch_size=32, epochs=50):
    """
    Train the sign language recognition model
    """
    # Load and preprocess data
    X, y = load_data(data_dir)
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize data
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    # Initialize model
    model = SignLanguageTranslator(
        input_shape=(64, 64, 3),
        num_classes=len(np.unique(np.argmax(y, axis=1)))
    )
    
    # Compile model
    model.compile_model()
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save model
    model.save_model(str(Path(model_dir) / "sign_language_model.h5"))
    
    return history

def train_text_to_sign_model(text_data_dir, model_dir, batch_size=32, epochs=50):
    """
    Train the text-to-sign language generation model
    """
    # Load text data
    # This is a placeholder - you'll need to implement text data loading
    # based on your specific data format
    X_train = np.random.rand(1000, 50)  # Placeholder
    y_train = np.random.randint(0, 1000, (1000, 50))  # Placeholder
    X_val = np.random.rand(200, 50)  # Placeholder
    y_val = np.random.randint(0, 1000, (200, 50))  # Placeholder
    
    # Initialize model
    model = TextToSignGenerator(
        vocab_size=1000,
        max_seq_length=50
    )
    
    # Compile model
    model.compile_model()
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save model
    model.save_model(str(Path(model_dir) / "text_to_sign_model.h5"))
    
    return history

if __name__ == "__main__":
    # Set paths
    data_dir = "data/processed/augmented"
    model_dir = "models"
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Train sign language recognition model
    print("Training sign language recognition model...")
    sign_language_history = train_sign_language_model(
        data_dir,
        model_dir,
        batch_size=32,
        epochs=50
    )
    
    # Train text-to-sign model
    print("Training text-to-sign model...")
    text_to_sign_history = train_text_to_sign_model(
        "data/text",  # Placeholder path
        model_dir,
        batch_size=32,
        epochs=50
    )
    
    print("Training completed!") 