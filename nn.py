import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

def main():
    # Model / data parameters
    num_classes = 6
    input_shape = (32, 32, 3)

    # Load the data and split it between train and test sets

    train_ds = keras.utils.image_dataset_from_directory(
        directory='ground_truth/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(32, 32))
    validation_ds = keras.utils.image_dataset_from_directory(
        directory='ground_truth/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(32, 32))

    #data_augmentation = keras.Sequential(
    #    [
    #        layers.RandomFlip("horizontal"),
    #        layers.RandomRotation(0.1),
    #    ]
    #)

    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

    model.summary()

    batch_size = 64
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_ds, batch_size=batch_size, epochs=epochs, validation_data = validation_ds)
    model.save('model.keras', overwrite=True)

if __name__ == '__main__':
    main()
