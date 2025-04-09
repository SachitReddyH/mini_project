import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class IndianSignLanguageDataset:
    def __init__(self, data_dir):
        """
        Initialize the Indian Sign Language dataset.
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = data_dir
        self.classes = []
        self.class_to_idx = {}
        self.image_paths = []
        self.labels = []
        
        # Load dataset if it exists
        if os.path.exists(data_dir):
            self._load_dataset()
        else:
            print(f"Dataset directory {data_dir} does not exist.")
            print("Creating directory structure for dataset collection.")
            self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the directory structure for the dataset."""
        # Create main data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create directories for each class (letter)
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            os.makedirs(os.path.join(self.data_dir, letter), exist_ok=True)
        
        print(f"Created directory structure in {self.data_dir}")
        print("Please add images to the respective class directories.")
    
    def _load_dataset(self):
        """Load the dataset from the data directory."""
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Sort classes alphabetically
        class_dirs.sort()
        
        # Set classes and class_to_idx
        self.classes = class_dirs
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        for cls_name in self.classes:
            cls_dir = os.path.join(self.data_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(cls_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Add image paths and labels
            for img_file in image_files:
                img_path = os.path.join(cls_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(cls_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {len(self.classes)} classes")
    
    def get_data_generators(self, validation_split=0.2, batch_size=32, img_size=(224, 224)):
        """
        Get data generators for training and validation.
        
        Args:
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for the generators
            img_size: Size to resize images to
            
        Returns:
            train_generator: Training data generator
            val_generator: Validation data generator
        """
        # Create data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Create data generator for validation (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def add_image(self, image, label, filename=None):
        """
        Add a new image to the dataset.
        
        Args:
            image: Image array (numpy array)
            label: Label for the image (string)
            filename: Optional filename for the image
        """
        # Check if label is valid
        if label not in self.class_to_idx:
            print(f"Invalid label: {label}")
            return False
        
        # Create filename if not provided
        if filename is None:
            import time
            filename = f"{label}_{int(time.time())}.jpg"
        
        # Create path for the image
        img_path = os.path.join(self.data_dir, label, filename)
        
        # Save the image
        cv2.imwrite(img_path, image)
        
        # Update dataset
        self.image_paths.append(img_path)
        self.labels.append(self.class_to_idx[label])
        
        print(f"Added image {filename} to class {label}")
        return True
    
    def get_dataset_info(self):
        """Get information about the dataset."""
        info = {
            "total_images": len(self.image_paths),
            "classes": self.classes,
            "class_counts": {}
        }
        
        # Count images per class
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            count = self.labels.count(cls_idx)
            info["class_counts"][cls_name] = count
        
        return info 