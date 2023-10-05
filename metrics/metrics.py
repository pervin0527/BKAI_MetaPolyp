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


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    focal_loss = - y_true * alpha * tf.math.pow((1. - y_pred), gamma) * tf.math.log(y_pred)
    focal_loss = tf.reduce_sum(focal_loss, axis=[1,2,3])
    return tf.reduce_mean(focal_loss)


def iou_loss(y_true, y_pred):
    iou_score = IoU(y_true, y_pred)
    return tf.reduce_mean(1 - iou_score)


def combined_loss(y_true, y_pred):
    loss_focal = focal_loss(y_true, y_pred) ## focal_loss(y_true[..., 1:], y_pred[..., 1:])
    loss_iou = iou_loss(y_true, y_pred) ## iou_loss(y_true[..., 1:], y_pred[..., 1:])
    focal_iou_loss = loss_focal + loss_iou

    cce = tf.keras.losses.CategoricalCrossentropy()
    multi_class_loss = cce(y_true, y_pred)

    loss = focal_iou_loss + multi_class_loss
    return loss
