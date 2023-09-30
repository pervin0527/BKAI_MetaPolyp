import cv2
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        fig, axes = plt.subplots(self.num_samples, 2, figsize=(10, 25))
        for i, idx in enumerate(sample_indices):
            file = pred_files[idx]
            image_path = f"{self.valid_dataset.image_dir}/{file}.jpeg"
            mask_path = f"{self.valid_dataset.mask_dir}/{file}.jpeg"
            
            image, mask = load_img_mask(image_path, mask_path, size=self.img_size)
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


def cosine_annealing_with_warmup(epochIdx):
    aMax, aMin = max_lr, min_lr
    warmupEpochs, stagnateEpochs, cosAnnealingEpochs = 0, 0, cos_anne_ep
    epochIdx = epochIdx % (warmupEpochs + stagnateEpochs + cosAnnealingEpochs)

    if(epochIdx < warmupEpochs):
        return aMin + (aMax - aMin) / (warmupEpochs - 1) * epochIdx
    else:
        epochIdx -= warmupEpochs
    if(epochIdx < stagnateEpochs):
        return aMax
    else:
        epochIdx -= stagnateEpochs

    return aMin + 0.5 * (aMax - aMin) * (1 + math.cos((epochIdx + 1) / (cosAnnealingEpochs + 1) * math.pi))


def plt_lr(step, schedulers):
    x = range(step)
    y = [schedulers(_) for _ in x]

    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()


def get_callbacks(monitor, mode, weight_path, log_path, _max_lr, _min_lr, _cos_anne_ep, save_weights_only):
    global max_lr
    max_lr = _max_lr
    global min_lr
    min_lr = _min_lr
    global cos_anne_ep
    cos_anne_ep = _cos_anne_ep

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                      patience=60,
                                                      restore_best_weights=True,
                                                      mode=mode)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                                     factor=0.1, ## 0.2
                                                     patience=10, ## 50
                                                     min_lr=1e-6, ## 1e-5
                                                     verbose=1,
                                                     mode=mode)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=weight_path,
                                                    monitor=monitor,
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=save_weights_only,
                                                    mode=mode,
                                                    save_freq="epoch")

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=0)

    csv_logger = tf.keras.callbacks.CSVLogger(f'{log_path}/training.csv')

    callbacks = [checkpoint, csv_logger, reduce_lr]
    return callbacks