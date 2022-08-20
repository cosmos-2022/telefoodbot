import keras
from PIL import Image
import numpy as np
import time
import os
from skimage import transform

MODEL_FOLDER = "Models"
IMAGE_PATH = 'infer_image.png'
#Classes = {'Apple': 0, 'Asparagus': 1, 'Aubergine': 2, 'Avocado': 3, 'Banana': 4, 'Brown-Cap-Mushroom': 5, 'Cabbage': 6, 'Carrots': 7, 'Cucumber': 8, 'Ginger': 9, 'Juice': 10, 'Kiwi': 11, 'Leek': 12, 'Lemon': 13, 'Lime': 14, 'Mango': 15, 'Melon': 16, 'Milk': 17, 'Oat-Milk': 18, 'Oatghurt': 19, 'Onion': 20, 'Orange': 21, 'Passion-Fruit': 22, 'Peach': 23, 'Pear': 24, 'Pepper': 25, 'Pineapple': 26, 'Pomegranate': 27, 'Potato': 28, 'Red-Beet': 29, 'Red-Grapefruit': 30, 'Satsumas': 31, 'Sour-Cream': 32, 'Soyghurt': 33, 'Tomato': 34, 'Yoghurt': 35, 'Zucchini': 36}
Classes = {'Apple': 0,
 'Asparagus': 1,
 'Aubergine': 2,
 'Avocado': 3,
 'Banana': 4,
 'Brown-Cap-Mushroom': 5,
 'Cabbage': 6,
 'Carrots': 7,
 'Cucumber': 8,
 'Garlic': 9,
 'Ginger': 10,
 'Juice': 11,
 'Kiwi': 12,
 'Leek': 13,
 'Lemon': 14,
 'Lime': 15,
 'Mango': 16,
 'Melon': 17,
 'Milk': 18,
 'Nectarine': 19,
 'Oat-Milk': 20,
 'Oatghurt': 21,
 'Onion': 22,
 'Orange': 23,
 'Pappaya': 24,
 'Passion-Fruit': 25,
 'Peach': 26,
 'Pear': 27,
 'Pepper': 28,
 'Pineapple': 29,
 'Plum': 30,
 'Pomegranate': 31,
 'Potato': 32,
 'Red-Beet': 33,
 'Red-Grapfruit': 34,
 'Satsumas': 35,
 'Sour-Cream': 36,
 'Sour-Milk': 37,
 'Soy-Milk': 38,
 'Soyghurt': 39,
 'Tomato': 40,
 'Yoghurt': 41,
 'Zucchini': 42}

Classes = {value : key for (key, value) in Classes.items()}

# Load the model
if 'mycnn' not in locals():
    model_path = os.path.join(MODEL_FOLDER, "cnn_model")
    mycnn = keras.models.load_model(model_path)

def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(filename, top_k=4):

    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)

    # Prediction
    proba = mycnn.predict(np_image)[0]
    np_ind = np.argsort(proba)
    # Sorted index list 
    indexes = sorted(range(len(proba)), key=lambda k: proba[k], reverse=True)

    # Get cuisine
    cusine_ordered = [Classes[ind] for ind in indexes]

    return cusine_ordered[:top_k]