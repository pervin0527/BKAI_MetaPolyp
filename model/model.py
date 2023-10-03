import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Conv2D
from keras.regularizers import L2
from keras_cv_attention_models import caformer
from model.layers import decode, convformer, merge, conv_bn_act

"""
    CAFormer,
    CAFormerS18,
    CAFormerS36,
    CAFormerM36,
    CAFormerB36,
    ConvFormerS18,
    ConvFormerS36,
    ConvFormerM36,
    ConvFormerB36,
"""

def build_model(img_size=256, num_classes=1, lambd=0.01):
    backbone = caformer.CAFormerS18(input_shape=(img_size, img_size, 3), pretrained="imagenet", num_classes=0)
    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers = [backbone.get_layer(x).output for x in layer_names]

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    x = layers[0]
    upscale_feature = decode(x, scale=4, filters=x.shape[channel_axis], lambd=lambd)

    for i, layer in enumerate(layers[1:]):
        x = decode(x, scale=2, filters=layer.shape[channel_axis], lambd=lambd)
        layer_fusion = convformer(layer, layer.shape[channel_axis], lambd=lambd)
        
        ## Doing multi-level concatenation
        if (i%2 == 1):
            upscale_feature = tf.keras.layers.Conv2D(layer.shape[channel_axis], (1, 1), activation="relu", padding="same", kernel_regularizer=L2(lambd))(upscale_feature)
            x = tf.keras.layers.Add()([x, upscale_feature])
            x = tf.keras.layers.Conv2D(x.shape[channel_axis], (1, 1), activation="relu", padding="same", kernel_regularizer=L2(lambd))(x)
        
        x = merge([x, layer_fusion], layer.shape[channel_axis])
        x = conv_bn_act(x, layer.shape[channel_axis], (1, 1), lambd=lambd)

        ## Upscale for next level feature
        if (i%2 == 1):
            upscale_feature = decode(x, scale = 8, filters = layer.shape[channel_axis], lambd=lambd)
        
    filters = x.shape[channel_axis] // 2
    upscale_feature = conv_bn_act(upscale_feature, filters, 1)
    x = decode(x, filters, 4, lambd=lambd)
    x = tf.keras.layers.Add()([x, upscale_feature])
    x = conv_bn_act(x, filters, 1)
    
    x = Conv2D(num_classes, kernel_size=1, padding='same', activation='softmax', kernel_regularizer=L2(lambd))(x)
    model = Model(backbone.input, x)

    return model
