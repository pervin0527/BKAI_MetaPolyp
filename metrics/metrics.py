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


def multi_class_focal_loss(y_true, y_pred, alpha=[0.25, 0.25, 0.25], gamma=2.0):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
    focal_loss_list = []
    for class_idx in range(y_pred.shape[-1]):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        alpha_t = y_true_class * alpha[class_idx] + (1 - y_true_class) * (1 - alpha[class_idx])
        focal_loss_class = - alpha_t * K.pow(1 - y_pred_class, gamma) * y_true_class * K.log(y_pred_class)
        
        focal_loss_list.append(K.sum(focal_loss_class))

    focal_loss = K.sum(focal_loss_list) / K.cast(K.shape(y_true)[0], 'float32')

    return focal_loss