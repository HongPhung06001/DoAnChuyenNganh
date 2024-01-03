import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def read_directory():
    for dirname, _, filenames in os.walk('DATA'):
        for filename in filenames:
            print(os.path.join(dirname, filename))



def load_dataset(net=True, dir=None):
    # Load and filter in Training/not Training data:
    df = pd.read_csv(dir)

    training = df.loc[(df['Usage'] == 'Training') & df['emotion'].isin([0, 1, 2, 3, 4, 5, 6])]
    testing = df.loc[(df['Usage'] != 'Training') & df['emotion'].isin([0, 1, 2, 3, 4, 5, 6])]

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_train]

    if net:
        X_train = [e.reshape((48, 48)).astype('float32') / 255 for e in X_train]
    else:
        X_train = [e.reshape((48, 48)) for e in X_train]
    X_train = np.array(X_train)
    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_test]
    if net:
        X_test = [e.reshape((48, 48)).astype('float32') / 255 for e in X_test]
    else:
        X_test = [e.reshape((48, 48)) for e in X_test]
    X_test = np.array(X_test)

    # y_train values:
    y_train = training[['emotion']].values
    y_train = keras.utils.to_categorical(y_train)

    # y_test values
    y_test = testing[['emotion']].values
    y_test = keras.utils.to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)

def augment_dataset(X_train, X_test, y_train, y_test, shuffle=False, channels=1):
    if shuffle:
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        X_train = X_train[indices]
        y_train = y_train[indices]

    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator()

    X_train = X_train.reshape(*X_train.shape, channels)
    X_test = X_test.reshape(*X_test.shape, channels)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=64)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=64)

    return train_generator, test_generator

