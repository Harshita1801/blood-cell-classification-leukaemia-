import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from PIL import Image

def get_lime_image(model, image):
    explainer = lime_image.LimeImageExplainer()
    image = Image.open(image).convert('RGB')  # PIL
    image_np = np.array(image)
    img = cv2.resize(image_np, (244, 244))  # Resize to match model input
    img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
    img = np.expand_dims(img_array, axis=0)

    explanation = explainer.explain_instance(
        image=img_arr.astype('double'),
        classifier_fn=lambda x: model.predict(x),
        top_labels=10,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False,
        num_features=15,
        min_weight=0.0
    )

    img_bound = mark_boundaries(temp / 255.0, mask)
    fig, ax = plt.subplots()
    ax.imshow(img_bound)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)
