import tensorflow as tf 
import keras.backend as K 
from keras.losses import categorical_crossentropy

def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    dice_score = tf.reduce_mean(dice_score)
    return dice_score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)

    return loss


def ce_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def IoU(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, dtype=tf.float32)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou_score = (intersection + epsilon) / (union + epsilon)

    return iou_score