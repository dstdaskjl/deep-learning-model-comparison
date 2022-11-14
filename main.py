import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping
from keras.activations import softmax, relu, sigmoid
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.layers import Flatten, UpSampling2D, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, BatchNormalization, Dropout, Conv2D, MaxPool2D, MaxPooling2D


def load_data():
    df = pd.read_csv('??', sep=',', header=None)
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randint(0, 99), shuffle=True)
    return x_train, x_test, y_train, y_test


def load_model(model: keras.applications, input_shape: list) -> keras.Sequential:
    model = keras.Sequential(
        [
            model(input_shape=input_shape, include_top=False),
            GlobalAveragePooling2D(),
            Dense(30, activation=relu),
            Dense(30, activation=relu),
            Dense(4, activation=softmax)
        ]
    )
    model.compile(optimizer=SGD(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    return model


x_train, x_test, y_train, y_test = load_data()
model = load_model(model=MobileNet, input_shape=x_train[0].shape)