from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import numpy as np
import cv2
import av
# from load_model import model
from keras.models import model_from_json
from PIL import Image
from Detection import processing, detection_with_img
from Detection.videoTransform import VideoTransformer
import streamlit as st
import base64

def render_svg(svg_file):
    with open(svg_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    svg = "".join(lines)
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)

#ánh xạ nhãn cảm xúc
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'sad', 5 : 'surprise', 6 : 'neutral'}

#khai báo đường của file json và h5
dir_json_fer = 'MODEL/emotiondetector_fer2013.json'
dir_h5_fer = 'MODEL/cnn_fer2013.h5'
dir_json_ck = 'MODEL/emotiondetector_ckextend.json'
dir_h5_ck = 'MODEL/cnn_ckextend.h5'


def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Chọn chế độ ứng dụng", ["NHẬN DẠNG", "XEM DỮ LIỆU"])

    if app_mode == 'XEM DỮ LIỆU':
        st.sidebar.subheader("DATA")
        dataset = st.sidebar.selectbox("Chọn bộ dữ liệu", ["fer2013", "ckextend"])
        st.sidebar.write(f"Bạn đã chọn bộ dữ liệu {dataset}")

        if dataset == 'fer2013':
            st.title("BỘ DỮ LIỆU FER2013")

            st.subheader('Biểu đồ mô tả dữ liệu fer2013')
            image = Image.open('VIEW_DATA/fer2013.png')
            st.image(image, caption='Minh họa từ bộ dữ liệu fer2013')

            st.subheader('Biểu đồ mô tả dữ liệu fer2013')
            image = Image.open('VIEW_DATA/View_image_fer2013.png')
            st.image(image, caption='Minh họa từ bộ dữ liệu fer2013')

            st.subheader('Biểu đồ độ chính xác và mấy mát trong quá trình huấn luyện')
            st.write('Bểu đồ sẽ trực quan hóa quá trình học của mô hình thông qua các biểu đồ mất mát và độ chính xác theo từng epoch.\
                        Điều này giúp chúng ta nắm bắt được cách mô hình học và thích ứng với dữ liệu huấn luyện,\
                        cũng như phát hiện các vấn đề tiềm ẩn như quá khớp hoặc chưa khớp')
            render_svg('RESULT/accuracy_loss_fer2013.svg')

            st.subheader('Độ chính xác của mô hình trên tập test')
            st.markdown("<h1 style='text-align: center; color: orange;'>Accuracy = 0.6734466425188075</h1>",
                        unsafe_allow_html=True)

            st.subheader('Biểu đồ ma trận nhầm lẫn trên tập test')
            render_svg('RESULT/confusion_matrix_fer2013_cm.svg')

        if dataset == 'ckextend':
            st.title("BỘ DỮ LIỆU CK+")

            st.subheader('Biểu đồ mô tả dữ liệu CK+')
            image = Image.open('VIEW_DATA/ckextended.png')
            st.image(image, caption='Biểu đồ mô tả dữ liệu ck+')

            st.subheader('Biểu đồ mô tả dữ liệu CK+')
            image = Image.open('VIEW_DATA/View_image_ckextended.png')
            st.image(image, caption='Minh họa từ bộ dữ liệu ck+')


            st.subheader('Biểu đồ độ chính xác và mấy mát trong quá trình huấn luyện')
            st.write('Bểu đồ sẽ trực quan hóa quá trình học của mô hình thông qua các biểu đồ mất mát và độ chính xác theo từng epoch.\
                        Điều này giúp chúng ta nắm bắt được cách mô hình học và thích ứng với dữ liệu huấn luyện,\
                        cũng như phát hiện các vấn đề tiềm ẩn như quá khớp hoặc chưa khớp')
            render_svg('RESULT/accuracy_loss_ckextend.svg')

            st.subheader('Độ chính xác của mô hình trên tập test')
            st.markdown("<h1 style='text-align: center; color: orange;'>Accuracy = 0.8846153846153846</h1>",
                       unsafe_allow_html=True)

            st.subheader('Biểu đồ ma trận nhầm lẫn trên tập test')
            render_svg('RESULT/confusion_matrix_ckextend_cm.svg')

    elif app_mode == 'NHẬN DẠNG':
        st.sidebar.subheader("NHẬN DẠNG")
        mode = st.sidebar.selectbox("Bạn muốn nhận dạng bằng ", ["Ảnh", "Camera"])

        if mode == "Ảnh":
            dataset = st.sidebar.selectbox("Nhận dạng với mô hình được huấn luyện bằng bộ dữ liệu", ["fer2013", "ckextend"])

            if dataset == "fer2013":
                st.sidebar.write(f'Bạn đã chọn nhận dạng bằng {mode} với bộ dữ liệu {dataset}')
                uploaded_file = st.file_uploader("Chọn một tệp hình ảnh", type=["jpg", "png"])

                if uploaded_file is not None:
                    st.image(uploaded_file, caption='Hình ảnh đã tải lên.')
                    st.write("Hình ảnh đã được tải lên thành công!")
                    if st.button('Nhận dạng'):
                        face_cascade, model = processing.load_model_to_detection(dir_json_fer,dir_h5_fer)
                        image = Image.open(uploaded_file)
                        detection_with_img.with_img(image, face_cascade, model, labels)

            if dataset == "ckextend":
                st.sidebar.write(f'Bạn đã chọn nhận dạng bằng {mode} với bộ dữ liệu {dataset}')
                uploaded_file = st.file_uploader("Chọn một tệp hình ảnh", type=["jpg", "png"])
                if uploaded_file is not None:
                    st.image(uploaded_file, caption='Hình ảnh đã tải lên.')
                    st.write("Hình ảnh đã được tải lên thành công!")
                    if st.button('Nhận dạng'):
                        face_cascade, model = processing.load_model_to_detection(dir_json_ck,dir_h5_ck)
                        image = Image.open(uploaded_file)
                        detection_with_img.with_img(image, face_cascade, model, labels)

        if mode == "Camera":
            dataset = st.sidebar.selectbox("Nhận dạng với mô hình được huấn luyện bằng bộ dữ liệu",
                                            ["fer2013", "ckextend"])

            if dataset == "fer2013":
                st.sidebar.write(f'Bạn đã chọn nhận dạng bằng {mode} với bộ dữ liệu {dataset}')
                face_cascade_camera, model_camera = processing.load_model_to_detection(dir_json_fer, dir_h5_fer)
                # video_processor = VideoTransformer(face_cascade_camera, model_camera, labels)
                def video_transformer_factory():
                    return VideoTransformer(face_cascade_camera, model_camera, labels)

                webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=video_transformer_factory,
                                         rtc_configuration=RTCConfiguration(
                                             {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}))

            if dataset == "ckextend":
                st.sidebar.write(f'Bạn đã chọn nhận dạng bằng {mode} với bộ dữ liệu {dataset}')
                face_cascade_camera, model_camera = processing.load_model_to_detection(dir_json_ck, dir_h5_ck)

                # video_processor = VideoTransformer(face_cascade_camera, model_camera, labels)
                def video_transformer_factory():
                    return VideoTransformer(face_cascade_camera, model_camera, labels)

                webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=video_transformer_factory,
                                             rtc_configuration=RTCConfiguration(
                                                 {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}))

if __name__ == "__main__":
    main()

