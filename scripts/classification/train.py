import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras, nn
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from .dataloader import ForestFireDataLoader

class ForestFireClassification():
    def __init__(self, forest_fire_dataloader : ForestFireDataLoader, class_names):
        self.forest_fire_dataloader = forest_fire_dataloader
        self.class_names = class_names
        self.model = self._model()

    def _model(self):
        return Sequential([
                layers.RandomFlip("horizontal", input_shape=(self.forest_fire_dataloader.dataset.img_height, self.forest_fire_dataloader.dataset.img_width, 3)),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                
                layers.Rescaling(1./255),

                layers.Conv2D(32, 7, padding='same', activation='relu'),
                layers.MaxPooling2D(pool_size=2, strides=2),
                
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(pool_size=2, strides=2),
                
                layers.Dropout(0.2),
                
                layers.Flatten(),
                
                layers.Dense(256, activation='relu'),
                layers.Dense(len(self.class_names))])

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def fit(self, epochs):
        return self.model.fit(self.forest_fire_dataloader.train_ds,
                              validation_data=self.forest_fire_dataloader.val_ds,
                              epochs=epochs)

    def show_training_results_plot(self, history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self, img_path):
        img = tf.keras.utils.load_img(
            img_path, target_size=(self.forest_fire_dataloader.dataset.img_height, self.forest_fire_dataloader.dataset.img_width)
        )

        img_array = tf.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        score = nn.softmax(predictions[0])

        return img, f"This image most likely belongs to {self.class_names[np.argmax(score)]} with a {(100 * np.max(score)):.2f} percent confidence."