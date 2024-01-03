import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import numpy as np
import cv2
import av
# from load_model import model
from keras.models import model_from_json
from PIL import Image
from Detection.processing import extract_features, load_model_to_detection

class VideoTransformer(VideoProcessorBase):
    def __init__(self, face_cascade, model, labels):
        self.face_cascade = face_cascade
        self.model = model
        self.labels = labels

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            face_image = gray[q:q+s, p:p+r]
            cv2.rectangle(image, (p, q), (p+r, q+s), (0, 225, 225), 2)
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = self.model.predict(img)
            prediction_label = self.labels[pred.argmax()]
            cv2.putText(image, '% s' % (prediction_label), (p-15, q-15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (0, 225, 0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")