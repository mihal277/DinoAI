import os

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class ModelBuilder:

    def __init__(self, game_wrapper, img_cols, img_rows, img_channels, actions, learning_rate):
        self.model = Sequential()
        self.model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4),
                         input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(actions))
        adam = Adam(lr=learning_rate)
        self.model.compile(loss='mse', optimizer=adam)

        if not os.path.isfile(game_wrapper.loss_file_path):
            self.model.save_weights('model.h5')
