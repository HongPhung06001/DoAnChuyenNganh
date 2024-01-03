from work_with_data import read_data
from model_cnn import model
read_data.read_directory()


(X_train, y_train) , (X_test, y_test) =read_data.load_dataset(dir='DATA/fer2013.csv')

train_generator, test_generator =read_data.augment_dataset(X_train, X_test, y_train, y_test, shuffle=False)

model = model.build_model_cnn()

from keras.optimizers import Adam



from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00025), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)

model_checkpoint = ModelCheckpoint('MODEL/cnn_fer2013.h5', save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=1)
# model_checkpoint = ModelCheckpoint('MODEL/cnn_ckextend.h5', save_best_only=True, save_weights_only=False, monitor='val_accuracy', mode='max', verbose=1)

history = model.fit(train_generator, epochs=100, validation_data=test_generator, callbacks=[early_stopping, model_checkpoint])



import pickle
# Lưu lịch sử huấn luyện
with open('MODEL/history_fer2013.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# with open('MODEL/history_ckextend.pkl', 'wb') as f:
#     pickle.dump(history.history, f)



model_json = model.to_json()
with open("MODEL/emotiondetector_fer2013.json", 'w') as json_file:
    json_file.write(model_json)

# with open("MODEL/emotiondetector_ckextend.json", 'w') as json_file:
#     json_file.write(model_json)


test_loss, test_accuracy = model.evaluate(test_generator)
print(f"TEST LOSS FUNCION: { test_loss }")
print(f"TEST ACCURACY: { test_accuracy }")



