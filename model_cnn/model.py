from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json



def build_model_cnn():
    new_model = Sequential()

    new_model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D((2, 2)))
    new_model.add(Dropout(0.25))

    new_model.add(Conv2D(128, (5, 5), padding='same'))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D((2, 2)))
    new_model.add(Dropout(0.25))

    new_model.add(Conv2D(512, (3, 3), padding='same'))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D((2, 2)))
    new_model.add(Dropout(0.25))

    new_model.add(Conv2D(512, (3, 3), padding='same'))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(MaxPooling2D((2, 2)))
    new_model.add(Dropout(0.25))

    new_model.add(Flatten())
    new_model.add(Dense(256))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.25))

    new_model.add(Dense(512))
    new_model.add(BatchNormalization())
    new_model.add(Activation('relu'))
    new_model.add(Dropout(0.25))

    new_model.add(Dense(7, activation='softmax'))
    return new_model

def load_model(model_json: object, model_weights: object) -> object:
    json_file = open(model_json, "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    return model
