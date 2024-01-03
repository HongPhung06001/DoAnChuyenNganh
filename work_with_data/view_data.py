import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt

import os

def view_data(dir, labels):
    #in ra hai biểu đồ biểu diễn dữ liệu fer2013 và ckextend trong VIEW_DATA
    df = pd.read_csv(dir)

    # Định nghĩa các nhãn
    label =  labels

    # Đếm số lượng các nhãn
    label_counts = df['emotion'].value_counts().sort_index().rename(index=label)
    # Vẽ biểu đồ với mỗi cột một màu
    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.index, label_counts.values, color=plt.cm.viridis(np.linspace(0, 1, len(label_counts))))
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Number of labels')

    # Trích xuất tên file từ đường dẫn
    base = os.path.basename(dir)
    filename = os.path.splitext(base)[0]

    plt.savefig(f'VIEW_DATA/{filename}.png')
    plt.show()

def view_image_data(dir, labels):
    #Hai ảnh View_image_fer2013 và View_image_ckextend trong VIEW_DATA
    df = pd.read_csv(dir)
    unique_labels = sorted(df.emotion.unique())
    num_labels = len(unique_labels)

    # Tính số hàng và cột cho subplot
    num_rows = num_labels
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 14))

    for i, label in enumerate(unique_labels):
        for j in range(num_cols):
            px = df[df.emotion == label].pixels.iloc[j]
            px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

            ax = axs[i, j]
            ax.imshow(px, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(labels[label])

    plt.tight_layout()

    base = os.path.basename(dir)
    filename = os.path.splitext(base)[0]
    plt.savefig(f'VIEW_DATA/View_image_{filename}.png')
    plt.show()
