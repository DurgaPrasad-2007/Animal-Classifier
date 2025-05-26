import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'animal_classifier.h5')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.pkl')

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def train_model():
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Load training data
    train_generator = datagen.flow_from_directory(
        '../Dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Load validation data
    validation_generator = datagen.flow_from_directory(
        '../Dataset',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Get number of classes
    num_classes = len(train_generator.class_indices)
    
    # Save class indices
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(train_generator.class_indices, f)

    # Create and train model
    model = create_model(num_classes)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved successfully at {MODEL_PATH}!")

if __name__ == "__main__":
    train_model() 