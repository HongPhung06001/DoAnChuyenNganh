import numpy as np
import cv2
from Detection.processing import load_model_to_detection, extract_features
from Detection.videoTransform import VideoTransformer
import streamlit as st


def with_img(image,face_cascade, model, labels):
    im = np.array(image)

    # Chuyển đổi hình ảnh sang grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Nhận dạng khuôn mặt
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    try:
        for (p, q, r, s) in faces:
            # Trích xuất khuôn mặt
            face = gray[q:q + s, p:p + r]

            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(im, (p, q), (p + r, q + s), (0, 225, 225), 2)

            # Điều chỉnh kích thước khuôn mặt và trích xuất đặc trưng
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)

            # Dự đoán nhãn cảm xúc
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Ghi nhãn cảm xúc lên hình ảnh
            cv2.putText(im, '% s' % (prediction_label), (p - 15, q - 15),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (0, 225, 0), 2)

        # Hiển thị hình ảnh đã được xử lý
        st.image(im, use_column_width=True)

    except cv2.error:
        pass