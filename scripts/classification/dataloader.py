from .dataset import ForestFireDataSet
import tensorflow as tf

class ForestFireDataLoader():
    def __init__(self, dataset : ForestFireDataSet, batch_size=32):
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_ds = None
        self.val_ds = None
        self.load_dataset()

    def load_dataset(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
                            self.dataset.train_data_dir,
                            validation_split=0.2,
                            subset="training",
                            seed=123,
                            image_size=(self.dataset.img_height, self.dataset.img_width),
                            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
                            self.dataset.train_data_dir,
                            validation_split=0.2,
                            subset="validation",
                            seed=123,
                            image_size=(self.dataset.img_height, self.dataset.img_width),
                            batch_size=self.batch_size)
    
    def autotune(self):
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE) 