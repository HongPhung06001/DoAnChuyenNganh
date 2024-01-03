from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_confusion_matrix(y_pred, y_test, labels, name, dir):
    # tính toán ma trận nhầm lẫn từ nhãn thực tế
    conf_matrix = confusion_matrix([np.argmax(y) for y in y_test], [np.argmax(y) for y in y_pred])

    # Tạo một hình ảnh để tìm ra sự nhầm lẫn trong ma trận
    plt.figure(figsize=(8, 6))

    # Sử dụng seaborn để tạo heatmap từ ma trận nhầm lẫn
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=labels, yticklabels=labels)

    # Đặt nhãn cho trục x, y và tiêu đề
    plt.xlabel('Nhãn dự đoán')
    plt.ylabel('Nhãn thật')
    plt.title('Ma trận nhầm lẫn')

    # Lưu và hiển thị biểu đồ
    plt.savefig(f'{dir}/confusion_matrix_{name}_cm.svg', format='svg')
    plt.show()