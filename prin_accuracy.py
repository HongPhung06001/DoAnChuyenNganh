from model_cnn import model, accuracy_and_loss, measurements, confusion_matrix
from keras.models import model_from_json
import pandas as pd
from work_with_data import read_data
import pickle
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv('DATA/fer2013.csv')
print("số dữ liệu", len(df))

labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
dir_result = 'RESULT'
def evaluate_and_visualize_mode_fer2013():
    name_fer = 'fer2013'
    print("---------------------load model---------------------")
    #tải mô hình
    fer2013 = model.load_model('MODEL/emotiondetector_fer2013.json', 'MODEL/cnn_fer2013.h5')

    #tải lại tập dữ liệu
    (X_train, y_train) , (X_test, y_test) =read_data.load_dataset(dir='DATA/fer2013.csv')
    train_generator, test_generator =read_data.augment_dataset(X_train, X_test, y_train, y_test, shuffle=False)

    print("---------------THỰC HIỆN DỰ ĐOÁN NHÃN TRÊN TẬP X_test----------------------")
    y_pred = fer2013.predict(X_test)

    # Tính toán và in ra độ chính xác của mô hình trên tập test
    # Chuyển đổi dự đoán và nhãn thực tế về dạng nhãn
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test_labels, y_pred_labels)

    print(f'Accuracy of model training with fer2013: {accuracy}')

    #in ra báo cáo phân loại với các độ đo f1-score, recall,...
    measurements.print_measurements(y_pred, y_test, labels, name = name_fer, dir = dir_result)

    #vẽ ma trận nhầm lẫn
    confusion_matrix.draw_confusion_matrix(y_pred, y_test, labels, name = name_fer, dir= dir_result)

    #vẽ biểu đồ độ chính xác và mất mát trong quán trình huấn luyện
    with open('MODEL/history_fer2013.pkl', 'rb') as f:
        history = pickle.load(f)
    accuracy_and_loss.draw_accuracy_loss(history, name_fer, dir_result)

def evaluate_and_visualize_mode_ckextend():
    name_ck = 'ckextend'

    #tải mô hình
    ckextend = model.load_model( 'MODEL/emotiondetector_ckextend.json','MODEL/cnn_ckextend.h5')

    #tải lại tập dữ liệu
    (X_train, y_train) , (X_test, y_test) =read_data.load_dataset(dir='DATA/ckextended.csv')
    train_generator, test_generator =read_data.augment_dataset(X_train, X_test, y_train, y_test, shuffle=False)

    print("---------------THỰC HIỆN DỰ ĐOÁN NHÃN TRÊN TẬP X_test----------------------")
    y_pred = ckextend.predict(X_test)

    # Tính toán và in ra độ chính xác của mô hình trên tập test
    # Chuyển đổi dự đoán và nhãn thực tế về dạng nhãn
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test_labels, y_pred_labels)

    print(f'Accuracy of model training with ck+: {accuracy}')

    #in ra báo cáo phân loại với các độ đo f1-score, recall,...
    measurements.print_measurements(y_pred, y_test, labels, name = name_ck, dir = dir_result)

    #vẽ ma trận nhầm lẫn
    confusion_matrix.draw_confusion_matrix(y_pred, y_test, labels, name = name_ck, dir= dir_result)

    #vẽ biểu đồ độ chính xác và mất mát trong  quá trình huấn luyện với ck+
    with open('MODEL/history_ckextend.pkl', 'rb') as f:
        history = pickle.load(f)
    accuracy_and_loss.draw_accuracy_loss(history, name_ck, dir_result)



evaluate_and_visualize_mode_fer2013()
evaluate_and_visualize_mode_ckextend()
