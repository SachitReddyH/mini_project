import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SignLanguageTranslator:
    def __init__(self, input_shape=(64, 64, 3), num_classes=100):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN for feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Reshape for LSTM
        x = layers.Reshape((-1, 128))(x)
        
        # LSTM layers for temporal processing
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.LSTM(128)(x)
        
        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save(path)

    @classmethod
    def load_model(cls, path):
        instance = cls()
        instance.model = tf.keras.models.load_model(path)
        return instance

class TextToSignGenerator:
    def __init__(self, vocab_size=1000, max_seq_length=50):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.model = self._build_model()

    def _build_model(self):
        # Input layer for text
        inputs = layers.Input(shape=(self.max_seq_length,))
        
        # Embedding layer
        x = layers.Embedding(self.vocab_size, 256)(inputs)
        
        # Transformer encoder
        for _ in range(6):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=8, key_dim=64
            )(x, x)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn_output = layers.Dense(1024, activation='relu')(x)
            ffn_output = layers.Dense(256)(ffn_output)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization()(x)
        
        # Output layer for sign sequence generation
        outputs = layers.Dense(self.vocab_size, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

    def generate_sign_sequence(self, text_input):
        # Convert text to sequence
        # This is a placeholder - you'll need to implement text preprocessing
        sequence = self._preprocess_text(text_input)
        
        # Generate sign sequence
        predictions = self.model.predict(sequence)
        return self._postprocess_predictions(predictions)

    def _preprocess_text(self, text):
        # Placeholder for text preprocessing
        # You'll need to implement tokenization and sequence padding
        pass

    def _postprocess_predictions(self, predictions):
        # Placeholder for converting model predictions to sign sequences
        # You'll need to implement conversion to actual sign language gestures
        pass

    def save_model(self, path):
        self.model.save(path)

    @classmethod
    def load_model(cls, path):
        instance = cls()
        instance.model = tf.keras.models.load_model(path)
        return instance 