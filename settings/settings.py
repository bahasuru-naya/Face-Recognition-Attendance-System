"""
Configuration settings for the face recognition system
"""
import os

# Camera settings
CAMERA = {
    'index': 0,  # Default camera (0 is usually built-in webcam)
    'width': 640,
    'height': 480
}

# Face detection settings
FACE_DETECTION = {
    'face_img_size': (224, 224)
}

# Training settings. Number of images needed to train the model.
TRAINING = {
    'samples_needed': 300
}

# File paths
PATHS = {
    'image_dir': 'registered_faces',
    'names_file': 'names.json',
    'model_file': 'models/trained_model.h5'

}

MEDIAPIPE_CONFIG = {
    'min_detection_confidence': 0.6, # Confidence threshold
    'padding': 25 # Pixels to add around the detected face box
}

RECOGNITION_CONFIG = {
    'confidence_threshold': 0.95, # Min confidence to recognize as known user
    'padding': 20, # Padding around detected face for cropping/display
    'model_input_size': (224, 224)
}
