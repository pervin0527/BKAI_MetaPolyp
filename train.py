import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import tensorflow as tf
import tensorflow_addons as tfa

from datetime import datetime
from model.model import build_model
from data.BKAIDataset import BKAIDataset
from data.BalancedBKAIDataset import BalancedBKAIDataset
from utils.callbacks import get_callbacks, SavePredictions
from metrics.metrics import dice_loss, dice_coefficient, ce_dice_loss, IoU, categorical_focal_loss, jaccard_loss

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

    save_dir = config["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        config["save_dir"] = save_path
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/preds")
        os.makedirs(f"{save_path}/logs")

    model = build_model(img_size=config["img_size"], num_classes=3)

    # train_dataset = BKAIDataset(config=config, split=config["train"])
    # valid_dataset = BKAIDataset(config=config, split=config["valid"])
    train_dataset = BalancedBKAIDataset(config=config, split=config["train"])
    valid_dataset = BalancedBKAIDataset(config=config, split=config["valid"])

    train_dataloader = tf.data.Dataset.from_generator(lambda: train_dataset, 
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32)))

    valid_dataloader = tf.data.Dataset.from_generator(lambda: valid_dataset, 
                                                      output_signature=(
                                                          tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(None, config["img_size"], config["img_size"], 3), dtype=tf.float32)))

    pred_callback = SavePredictions(model, valid_dataset, save_dir=f"{save_path}/preds", num_samples=config["num_pred_samples"])
    callbacks = get_callbacks(monitor='val_loss',
                              mode = 'min',
                              weight_path=f"{save_path}/weights/ckpt",
                              log_path=f"{save_path}/logs", 
                              _max_lr=config["max_lr"],
                              _min_lr=config["min_lr"],
                              _cos_anne_ep = 1000, 
                              save_weights_only=config["save_weights_only"])
    callbacks.append(pred_callback)
    

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(config["initial_lr"],
                                                                     config["decay_steps"],
                                                                     config["last_lr"],
                                                                     power=0.2)
    opts = tfa.optimizers.AdamW(learning_rate=config["initial_lr"], weight_decay=learning_rate_fn)

    # tf.keras.utils.get_custom_objects().update({"focal": categorical_focal_loss})
    # model.compile(optimizer=opts, loss='focal', metrics=[dice_coefficient, ce_dice_loss, IoU])

    # tf.keras.utils.get_custom_objects().update({"jacard": jaccard_loss})
    # model.compile(optimizer=opts, loss='jacard', metrics=[dice_coefficient, ce_dice_loss, IoU])

    tf.keras.utils.get_custom_objects().update({"dice": dice_loss})
    model.compile(optimizer=opts, loss='dice', metrics=[dice_coefficient, ce_dice_loss, IoU])

    history = model.fit(train_dataloader, 
                        epochs=config["epochs"],
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataset) // config["batch_size"],
                        validation_data=valid_dataloader,
                        validation_steps=len(valid_dataset) // config["batch_size"])
    
    model.save(f"{save_path}/weights/final_model.h5")