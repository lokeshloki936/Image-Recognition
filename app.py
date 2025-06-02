import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
import cv2 # OpenCV for image processing

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your pre-trained Keras model
MODEL_PATH = 'models/my_image_classifier_model.h5'
model = None # Initialize model to None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load class labels (e.g., 'cat', 'dog', 'airplane')
LABELS_PATH = 'models/labels.txt'
class_labels = [] # Initialize to empty list
try:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r') as f:
            class_labels = [line.strip() for line in f]
        print(f"Labels loaded successfully from {LABELS_PATH}")
    else:
        print(f"Error: Labels file not found at {LABELS_PATH}. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading labels: {e}")


# Define the target image size for your model
# IMPORTANT: This must match the input size your model was trained on!
# For the CIFAR-10 model trained in train_model.py, this is 32x32.
TARGET_SIZE = (32, 32)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size):
    """
    Loads an image using OpenCV, resizes it, converts to RGB, and prepares it for the model.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}. Check file path and permissions.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR (OpenCV default) to RGB
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0) # Add batch dimension (1, height, width, channels)
    # Important: Normalize image pixels (0-1) if your model was trained with normalization
    img = img / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url) # No file part
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url) # No selected file

    if file and allowed_file(file.filename):
        if model is None or not class_labels:
            # Render a specific error if model or labels aren't loaded
            return render_template('result.html', prediction="Error: Model or labels not loaded. Please ensure train_model.py ran successfully.", image_path=None)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Preprocess the image for the model
            processed_img = preprocess_image(filepath, TARGET_SIZE)

            # Make prediction
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100

            # Get the predicted label from our loaded class names
            if 0 <= predicted_class_index < len(class_labels):
                predicted_label = class_labels[predicted_class_index]
            else:
                predicted_label = f"Unknown Class Index {predicted_class_index}"

            prediction_text = f"{predicted_label} ({confidence:.2f}%)"

            # Pass the relative path for the HTML to display the image
            display_image_path = os.path.join('uploads', filename) # This is relative to static/

            return render_template('result.html', prediction=prediction_text, image_path=display_image_path)
        except Exception as e:
            # Catch any errors during preprocessing or prediction
            print(f"Prediction processing error: {e}")
            display_image_path = os.path.join('uploads', filename) if filename else None
            return render_template('result.html', prediction=f"An error occurred during prediction: {e}", image_path=display_image_path)
    return redirect(url_for('index')) # If file not allowed or other issue

if __name__ == '__main__':
    # Ensure the 'models' directory exists for app.py to try loading from
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(debug=True) # debug=True for development, set to False for production