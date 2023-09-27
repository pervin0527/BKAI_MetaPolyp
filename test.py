import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from model.model import build_model
from utils import decode_mask

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


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]

    return rle_to_string(rle)


def rle2mask(mask_rle, shape=(3,3)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)

    r = {'ids': ids, 'strings': strings,}

    return r


def test(model, test_files, save_dir):
    for test_file in test_files:
        file_name = test_file.split('/')[-1].split('.')[0]
        image = cv2.imread(test_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:-1]

        x = cv2.resize(image, (256, 256))
        x = (x / 127.5) - 1
        x = np.expand_dims(x, 0)

        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.argmax(y_pred, axis=-1)
        decoded_mask = decode_mask(y_pred)

        decoded_mask = cv2.resize(decoded_mask, (width, height))
        decoded_mask = cv2.cvtColor(decoded_mask, cv2.COLOR_BGR2RGB)

        cv2.imwrite(f"{save_dir}/{file_name}.png", decoded_mask)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    weight_dir = config["weight_dir"]
    save_dir = '/'.join(weight_dir.split('/')[:-2]) + "/test_result"
    print(save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model = build_model(img_size=config["img_size"], num_classes=3)
    model.load_weights(weight_dir)

    data_dir = config["data_dir"]
    test_set_name = config["test"]
    test_files = sorted(glob(f"{data_dir}/{test_set_name}/*"))
    print(len(test_files))

    test(model, test_files, save_dir)

    result = mask2string(save_dir)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = result['ids']
    df['Expected'] = result['strings']

    df.to_csv(r'output.csv', index=False)