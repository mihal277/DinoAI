from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tarfile
import numpy as np
import cv2
from math import ceil

if __name__ == '__main__':
    start_file = 0
    files_count = 33

    x = np.concatenate([np.load("output/X_" + str(i) + ".npz")['arr_0'] for i in range(start_file, start_file + files_count)])
    y = np.concatenate([np.load("output/y_" + str(i) + ".npz")['arr_0'] for i in range(start_file, start_file + files_count)])

    print(x.shape, y.shape)


    def preprocess_image(image):
        image = image[:, :300, :500]
        image = np.stack([cv2.resize(layer, (80,80)) for layer in image], axis=2)
        return image

    x = np.array([preprocess_image(img) for img in x])


    def split_into_classes(ds, y):
        indices_1 = [i for i, elem in enumerate((y == 1)) if elem == 1]
        indices_0 = [i for i, elem in enumerate((y == 0)) if elem == 1]
        return ds[indices_0], ds[indices_1]
                

    def get_balanced_dataset(x, y):
        x0, x1 = split_into_classes(x,y)
        y0, y1 = split_into_classes(y,y)

        clone_rate = ceil(len(y0) / (len(y1)))
        new_x1 = np.concatenate([x1 for _ in range(clone_rate)])
        new_y1 = np.concatenate([y1 for _ in range(clone_rate)])

        return np.concatenate([x0, new_x1]), \
                np.concatenate([y0, new_y1])

    x, y = get_balanced_dataset(x, y)
    print(x.shape, y.shape)

    y0, y1 = split_into_classes(y ,y)
    print(len(y0), len(y1))



    x_train, x_test, y_train, y_test = train_test_split(x, to_categorical(y), test_size=0.1)

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    print(np.sum(y_train[:, 0] == 1))
    print(np.sum(y_test[:, 0] == 1))


    im_w, im_h = 80, 80

    model = Sequential()

    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(im_w, im_h, 4)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(Conv2D(256, (4, 4), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))


    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)

    model.save('model.h5')
