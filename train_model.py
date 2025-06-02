import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Load and Preprocess Dataset (CIFAR-10 Example) ---
print("Loading CIFAR-10 dataset...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names (for display and for labels.txt)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Number of classes: {len(class_names)}")

# --- 2. Build the Convolutional Neural Network (CNN) Model ---
print("\nBuilding CNN model...")
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) # 10 classes for CIFAR-10

# Display the model's architecture
model.summary()

# --- 3. Compile the Model ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Use from_logits=False if your last layer is softmax
              metrics=['accuracy'])

# --- 4. Train the Model ---
print("\nStarting model training...")
# You can adjust epochs and batch_size based on your computational power and desired accuracy
history = model.fit(train_images, train_labels, epochs=10, # Train for 10 epochs
                    validation_data=(test_images, test_labels))
print("Model training finished.")

# --- 5. Save the Trained Model ---
# Create the 'models' directory if it doesn't exist
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_filename = os.path.join(models_dir, 'my_image_classifier_model.h5')
model.save(model_filename)
print(f"\nModel saved successfully to: {model_filename}")

# --- 6. Save Class Labels ---
labels_filename = os.path.join(models_dir, 'labels.txt')
with open(labels_filename, 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print(f"Class labels saved successfully to: {labels_filename}")

print("\nTraining script completed. You can now run 'app.py'.")

# IMPORTANT NOTE for Flask app:
# The input shape for your Flask app's preprocess_image function (TARGET_SIZE)
# MUST match the input shape your model was trained on.
# For this CIFAR-10 example, the input shape is (32, 32, 3), so TARGET_SIZE in app.py should be (32, 32).