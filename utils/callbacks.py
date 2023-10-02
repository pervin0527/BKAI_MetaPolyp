import cv2
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K
from utils.utils import decode_mask
from data.batch_preprocess import load_img_mask, normalize

class SavePredictions(tf.keras.callbacks.Callback):
    def __init__(self, model, valid_dataset, img_size, save_dir, num_samples=5):
        super(SavePredictions, self).__init__()
        self.model = model
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.img_size = img_size
        self.valid_dataset = valid_dataset

    def on_epoch_end(self, epoch, logs=None):
        pred_files = self.valid_dataset.file_list
        random.shuffle(pred_files)
        
        sample_indices = random.sample(range(len(pred_files)), self.num_samples)
        fig, axes = plt.subplots(self.num_samples, 2, figsize=(10, 5 * self.num_samples))
        for i, idx in enumerate(sample_indices):
            file = pred_files[idx]
            image_path = f"{self.valid_dataset.image_dir}/{file}.jpeg"
            mask_path = f"{self.valid_dataset.mask_dir}/{file}.jpeg"
            
            image, mask = load_img_mask(image_path, mask_path, size=self.img_size)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            x = normalize(image)
            x = np.expand_dims(x, 0)

            prediction = self.model.predict(x, verbose=0)[0]
            argmax_mask = np.argmax(prediction, axis=-1)
            decoded_mask = decode_mask(argmax_mask)

            original_mask = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
            pred_mask = cv2.addWeighted(image, 0.7, decoded_mask, 0.3, 0)

            original_mask = cv2.resize(original_mask, (width, height))
            pred_mask = cv2.resize(pred_mask, (width, height))
            
            axes[i, 0].imshow(original_mask)
            axes[i, 0].set_title("Original Mask")

            axes[i, 1].imshow(pred_mask)
            axes[i, 1].set_title("Predict Mask")

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/epoch_{epoch:>04}.png")
        plt.close()

class PrintLearningRate(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        print(f"\nEpoch {epoch+1}: Current Learning Rate: {current_lr:.10f}")


def get_callbacks(config, model, dataset):
    save_path = config["save_dir"]
    pred_callback = SavePredictions(model, dataset, img_size=config["img_size"], save_dir=f"{save_path}/preds", num_samples=config["num_pred_samples"])

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=config["early_stopping_patience"],
                                                      restore_best_weights=True,
                                                      monitor="val_loss",
                                                      mode="min",
                                                      verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=config["factor"], ## 0.2
                                                     patience=config["patience"], ## 50
                                                     min_lr=config["min_lr"], ## 1e-5
                                                     monitor="val_loss",
                                                     mode="min",
                                                     verbose=1,)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f"{save_path}/weights/best_weight.h5",
                                                    save_best_only=True,
                                                    save_weights_only=config["save_weights_only"],
                                                    monitor="val_loss",
                                                    save_freq="epoch",
                                                    mode="min",
                                                    verbose=1,)


    csv_logger = tf.keras.callbacks.CSVLogger(f'{save_path}/logs/train_log.csv')

    lr_printer = PrintLearningRate()

    callbacks = [pred_callback, lr_printer, checkpoint, csv_logger, early_stopping, reduce_lr]
    return callbacks