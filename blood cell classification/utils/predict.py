import numpy as np
from tensorflow.keras.models import load_model
from utils.lime_explainer import get_lime_image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = load_model('model/trained_model.h5')
class_dict = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']

def preprocess_image(uploaded_file):
    # Read image using PIL
    image = Image.open(uploaded_file).convert('RGB')
    
    # Convert to numpy array
    img_arr = np.array(image)

    # Resize using OpenCV
    img_resized = cv2.resize(img_arr, (244, 244))

    # Preprocess for model
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    return img_expanded


def predict_and_explain(uploaded_file):
    img_array = preprocess_image(uploaded_file)

    probs = model.predict(img_array)
    pred_index = np.argmax(probs[0])
    pred_class = class_dict[pred_index]
    probability = probs[0][pred_index] * 100

    lime_img = get_lime_image(model, uploaded_file)  # <-- Corrected this line

    return pred_class, probability, lime_img