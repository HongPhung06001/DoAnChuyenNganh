import matplotlib.pyplot as plt

def draw_accuracy_loss(history_dict, name, dir):
    training_accuracy = history_dict['accuracy']
    validation_accuracy = history_dict['val_accuracy']

    training_loss = history_dict['loss']
    validation_loss = history_dict['val_loss']

    #tạo một hình mới với kích thước 6x8
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, label='Training Accuracy')
    plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
    plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{dir}/accuracy_loss_{name}.svg', format='svg')
    plt.show()