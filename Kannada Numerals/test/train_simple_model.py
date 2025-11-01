import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_train_model():
    """Load data and train a simple CNN model"""
    
    # Load data from your Google Drive path
    data_path = "G:\\My Drive\\digit recognition\\train.csv"
    
    try:
        print("Loading data...")
        data = pd.read_csv(data_path)
        print(f"Data loaded: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using built-in MNIST dataset as fallback...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Combine and create DataFrame-like structure
        x_combined = np.concatenate([x_train, x_test])
        y_combined = np.concatenate([y_train, y_test])
        
        # Flatten images and create DataFrame
        pixels = x_combined.reshape(-1, 784)
        data = pd.DataFrame(pixels, columns=[f'pixel{i}' for i in range(784)])
        data['label'] = y_combined
    
    # Preprocess data
    labels = data['label'].values
    pixels = data.drop('label', axis=1).values
    
    # Normalize and reshape
    pixels = pixels.astype('float32') / 255.0
    pixels = pixels.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels, 10)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        pixels, labels, test_size=0.2, random_state=42
    )
    
    # Create simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model
    model.save('digit_recognition_cnn.h5')
    print("Model saved as 'digit_recognition_cnn.h5'")
    
    return model, history

if __name__ == "__main__":
    model, history = load_and_train_model()