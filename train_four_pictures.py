import argparse
from math import ceil
import tarfile

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


IM_W, IM_H = 80, 80

def preprocess_image(image):
    image = image[:, :300, :500]
    image = np.stack([cv2.resize(layer, (80,80)) for layer in image], axis=2)
    return image

def split_into_classes(ds, y):
    indices_1 = [i for i, elem in enumerate((y == 1)) if elem == 1]
    indices_0 = [i for i, elem in enumerate((y == 0)) if elem == 1]
    return ds[indices_0], ds[indices_1]
              

def get_balanced_dataset(x, y):
    x0, x1 = split_into_classes(x,y)
    y0, y1 = split_into_classes(y,y)

    clone_rate = ceil(len(y0) / len(y1))
    new_x1 = np.concatenate([x1 for _ in range(clone_rate)])
    new_y1 = np.concatenate([y1 for _ in range(clone_rate)])

    return np.concatenate([x0, new_x1]), \
            np.concatenate([y0, new_y1])

def load_dataset(path, files_count):
    x = np.concatenate([np.load(path + "/X_" + str(i) + ".npz")['arr_0'] for i in range(files_count)])
    y = np.concatenate([np.load(path + "/y_" + str(i) + ".npz")['arr_0'] for i in range(files_count)])

    x = np.array([preprocess_image(img) for img in x])

    x, y = get_balanced_dataset(x, y)

    return x, y

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(IM_W, IM_H, 4)))
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

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=[auroc, 'accuracy'])
    model.summary()

    return model

def train(model, x, y, epochs):
    x_train, x_test, y_train, y_test = train_test_split(x, to_categorical(y),\
        test_size=0.1, random_state=42)

    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_test, y_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_file', '-o')
    parser.add_argument('--dataset_path', '-d')
    parser.add_argument('--files_count', '-n', type=int)
    parser.add_argument('--input_model', '-i', required=False)
    parser.add_argument('--epochs', '-e', type=int)

    args = parser.parse_args()

    x, y = load_dataset(args.dataset_path, args.files_count)

    if args.input_model is not None:
        model = load_model(args.input_model)
    else:
        model = build_model()

    train(model, x, y, args.epochs)

    model.save(args.output_file)
