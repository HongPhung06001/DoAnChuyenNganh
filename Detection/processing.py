from keras.models import model_from_json
import cv2
import numpy as np

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def load_model_to_detection(dir_json, dir_h5):
    json_file = open(dir_json, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights(dir_h5)
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    return face_cascade, model