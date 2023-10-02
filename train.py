import os
import warnings
warnings.filterwarnings("ignore")
os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import segmentation_models as sm

from datetime import datetime
from data.BKAIDataset import BKAIDataset
from data.BalancedBKAIDataset import BalancedBKAIDataset

from model.model import build_model
from utils.callbacks import get_callbacks
from utils.utils import save_config_to_yaml
from metrics.metrics import dice_loss, dice_coefficient, ce_dice_loss, IoU

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = build_model(img_size=config["img_size"], num_classes=3, lambd=config["l2_lambd"])
    # model.summary()
    if config["weight_dir"] != "":
        model.load_weights(config["weight_dir"], by_name=True, skip_mismatch=True)

        for layer in model.layers:
            if 'stack' in layer.name:
                layer.trainable = False

    # train_dataset = BKAIDataset(config=config, split=config["train"])
    # valid_dataset = BKAIDataset(config=config, split=config["valid"])
    train_dataset = BalancedBKAIDataset(config=config, split=config["train"])
    valid_dataset = BalancedBKAIDataset(config=config, split=config["valid"])

    train_dataloader = tf.data.Dataset.from_generator(lambda: train_dataset, 
                                                      output_signature=(tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32)))

    valid_dataloader = tf.data.Dataset.from_generator(lambda: valid_dataset, 
                                                      output_signature=(tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32),
                                                                        tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32)))
    
    train_steps_per_epoch = len(train_dataset) // config["batch_size"]
    valid_steps_per_epoch = len(valid_dataset) // config["batch_size"]
    total_steps = train_steps_per_epoch * config["epochs"]
    warmup_steps = total_steps // 2

    save_dir = config["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        config["save_dir"] = save_path
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/preds")
        os.makedirs(f"{save_path}/logs")

    save_config_to_yaml(config, save_path)

    # learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=config["initial_lr"],
    #                                                                  decay_steps=config["decay_steps"],
    #                                                                  end_learning_rate=config["last_lr"],
    #                                                                  power=0.2)
    # optimizer = tfa.optimizers.AdamW(learning_rate=config["initial_lr"], weight_decay=learning_rate_fn)

    # tf.keras.utils.get_custom_objects().update({"dice": dice_loss})
    
    # callbacks = get_callbacks(config, model, dataset=valid_dataset)
    # model.compile(optimizer=optimizer, loss='dice', metrics=[dice_coefficient, ce_dice_loss, IoU])

    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=config["initial_lr"],
                                                             decay_steps=total_steps,
                                                             alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay)
    total_loss = sm.losses.DiceLoss(class_indexes=[1, 2]) + sm.losses.CategoricalFocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"], class_indexes=[1, 2])

    callbacks = get_callbacks(config, model, dataset=valid_dataset)
    callbacks.pop()
    model.compile(optimizer=optimizer, loss=total_loss, metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()])
    
    history = model.fit(train_dataloader, 
                        epochs=config["epochs"],
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=valid_dataloader,
                        validation_steps=valid_steps_per_epoch)
    
    model.save(f"{save_path}/weights/final_model.h5")