
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

#   custom configuration
class ModelConfig:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 100
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 0.001
    RANDOM_SEED = 42

config = ModelConfig()

def load_and_explore_datasets():
    """Load and analyze the Kannada MNIST datasets"""
    train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
    dig_data = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
    test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
    
    print(f"Training samples: {train_data.shape[0]}, Features: {train_data.shape[1]}")
    print(f"Dig-MNIST samples: {dig_data.shape[0]}")
    print(f"Test samples: {test_data.shape[0]}")
    
    return train_data, dig_data, test_data

def visualize_sample_digits(features, labels, sample_count=12):
    """Display sample Kannada digits with labels"""
    plt.figure(figsize=(16, 8))
    for index in range(sample_count):
        plt.subplot(3, 4, index + 1)
        digit_image = features[index].reshape(config.IMG_HEIGHT, config.IMG_WIDTH)
        plt.imshow(digit_image, cmap='viridis')
        plt.title(f'Digit: {labels[index]}', fontsize=12, pad=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def prepare_features_labels(dataframe, is_test_set=False):
    """Preprocess and prepare features and labels"""
    if not is_test_set:
        target_labels = dataframe['label'].values
        feature_values = dataframe.drop('label', axis=1).values
    else:
        target_labels = None
        feature_values = dataframe.drop('id', axis=1).values if 'id' in dataframe.columns else dataframe.values
    
    # normalize and reshape features
    feature_values = feature_values.astype('float32') / 255.0
    feature_values = feature_values.reshape(-1, config.IMG_HEIGHT, config.IMG_WIDTH, 1)
    
    if not is_test_set:
        target_labels = tf.keras.utils.to_categorical(target_labels, config.NUM_CLASSES)
        return feature_values, target_labels
    return feature_values

def create_custom_augmentation():
    """Create custom data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomRotation(factor=0.08, fill_mode='constant'),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode='constant'),
        layers.RandomTranslation(height_factor=0.12, width_factor=0.12, fill_mode='constant'),
    ])

def build_custom_cnn_model():
    """Construct custom CNN architecture for Kannada digit recognition"""
    input_layer = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1))
    
    # augmentation
    augmented = create_custom_augmentation()(input_layer)
    
    # first convolution block
    x = layers.Conv2D(36, (3, 3), activation='relu', padding='same')(augmented)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(36, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.28)(x)
    
    # Second convolution block
    x = layers.Conv2D(72, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(72, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.28)(x)
    
    # Third convolution block
    x = layers.Conv2D(144, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    x = layers.Flatten()(x)
    x = layers.Dense(312, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.52)(x)
    x = layers.Dense(156, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.52)(x)
    output_layer = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def setup_training_callbacks():
    """Configure training callbacks"""
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=16,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.55,
            patience=6,
            min_lr=1.2e-7,
            verbose=1,
            min_delta=0.002
        ),
        callbacks.ModelCheckpoint(
            'optimized_kannada_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

def plot_training_progress(history):
    """Visualize training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Progress', fontsize=14, pad=20)
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Progress', fontsize=14, pad=20)
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_predictions(model, features, labels, num_examples=8):
    """Evaluate and visualize model predictions"""
    plt.figure(figsize=(16, 8))
    predictions = model.predict(features[:num_examples])
    
    for i in range(num_examples):
        plt.subplot(2, 4, i + 1)
        plt.imshow(features[i].reshape(config.IMG_HEIGHT, config.IMG_WIDTH), cmap='plasma')
        
        true_label = np.argmax(labels[i])
        pred_label = np.argmax(predictions[i])
        confidence = np.max(predictions[i])
        
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.3f}', 
                 color=color, fontsize=11, pad=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# main execution pipeline
def main():
    # Load datasets
    train_df, dig_df, test_df = load_and_explore_datasets()
    
    # Display data information
    print("\nTraining label distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    # Visualize samples
    train_labels_raw = train_df['label'].values
    train_features_raw = train_df.drop('label', axis=1).values
    visualize_sample_digits(train_features_raw[:12], train_labels_raw[:12])
    
    # Preprocess data
    X_train, y_train = prepare_features_labels(train_df)
    X_dig, y_dig = prepare_features_labels(dig_df)
    X_test = prepare_features_labels(test_df, is_test_set=True)
    
    # Combine datasets
    X_combined = np.concatenate([X_train, X_dig])
    y_combined = np.concatenate([y_train, y_dig])
    
    # Split into training and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_combined, y_combined, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=config.RANDOM_SEED, 
        stratify=np.argmax(y_combined, axis=1)
    )
    
    print(f"\nFinal training set: {X_train_final.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Build and compile model
    kannada_model = build_custom_cnn_model()
    
    kannada_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    kannada_model.summary()
    
    # Train model
    training_history = kannada_model.fit(
        X_train_final, y_train_final,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=setup_training_callbacks(),
        verbose=1
    )
    
    # Load best model and evaluate
    optimized_model = tf.keras.models.load_model('optimized_kannada_model.h5')
    validation_loss, validation_accuracy = optimized_model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n Optimized Model Validation Accuracy: {validation_accuracy:.4f}")
    print(f"📉 Optimized Model Validation Loss: {validation_loss:.4f}")
    
    # Plot training history
    plot_training_progress(training_history)
    
    # Generate predictions
    test_predictions = optimized_model.predict(X_test)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_predicted_labels)),
        'label': test_predicted_labels
    })
    
    submission_df.to_csv('enhanced_kannada_submission.csv', index=False)
    print("\n Enhanced submission file created!")
    
    # Save final model
    optimized_model.save('final_kannada_digit_model.h5')
    print(" Model saved as 'final_kannada_digit_model.h5'")
    
    # Test predictions visualization
    evaluate_predictions(optimized_model, X_val, y_val)

if __name__ == "__main__":
    main()