import tensorflow as tf
import keras.backend as K
from keras.layers import Conv2D
from keras.regularizers import L2

def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), activation='relu', padding='same', lambd=0.05):
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=L2(lambd))(inputs)
    x = bn_act(x, activation=activation)
    return x


def merge(l, filters=None):
    if filters is None:
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = l[0].shape[channel_axis]
    
    x = tf.keras.layers.Add()([l[0],l[1]])
    
    # x = block(x, filters)
    
    return x


def bn_act(inputs, activation='swish'):
    x = tf.keras.layers.BatchNormalization()(inputs)
    if activation:
        x = tf.keras.layers.Activation(activation)(x)
    
    return x


def decode(input_tensor, filters, scale=2, activation='relu', lambd=0.01):
    x1 = tf.keras.layers.Conv2D(filters, (1, 1), activation=activation, use_bias=False, kernel_initializer='he_normal', padding='same', kernel_regularizer=L2(lambd))(input_tensor)
    x2 = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, use_bias=False, padding='same', kernel_regularizer=L2(lambd))(input_tensor)
    
    merge = tf.keras.layers.Add()([x1, x2])
    x = tf.keras.layers.UpSampling2D((scale, scale))(merge)

    skip_feature = tf.keras.layers.Conv2D(filters, (3, 3), activation=activation, use_bias=False, kernel_initializer='he_normal', padding='same', kernel_regularizer=L2(lambd))(merge)
    skip_feature = tf.keras.layers.Conv2D(filters, (1, 1), activation=activation, use_bias=False, kernel_initializer='he_normal', padding='same', kernel_regularizer=L2(lambd))(skip_feature)
    
    merge = tf.keras.layers.Add()([merge, skip_feature])

    x = bn_act(x, activation = activation)
    
    return x


def convformer(input_tensor, filters, padding="same", lambd=0.01):
    x = tf.keras.layers.LayerNormalization()(input_tensor)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size=(3,3), padding=padding, kernel_regularizer=L2(lambd))(x)
    # x = x1 + x2 + x3
    x = tf.keras.layers.Attention()([x, x, x])
    out = tf.keras.layers.Add()([x, input_tensor])
    
    x1 = tf.keras.layers.Dense(filters, activation="gelu", kernel_regularizer=L2(lambd))(out)
    x1 = tf.keras.layers.Dense(filters, kernel_regularizer=L2(lambd))(x1)
    out_tensor = tf.keras.layers.Add()([out, x1])
    return out_tensor