import os
import cv2
from tqdm import tqdm
from glob import glob

def save_overlap(image_files, mask_files):
    for idx in tqdm(range(len(image_files))):
        image_file, mask_file = image_files[idx], mask_files[idx]
        file_name = image_file.split('/')[-1].split('.')[0]

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (256, 256))

        overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{save_dir}/{file_name}.jpeg", overlay)


if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    image_dir = f"{data_dir}/train"
    mask_dir = f"{data_dir}/train_mask"

    image_files = sorted(glob(f"{image_dir}/*.jpeg"))
    mask_files = sorted(glob(f"{mask_dir}/*.jpeg"))

    save_dir = "./dataset"
    if not os.path.isdir("./dataset"):
        os.makedirs("./dataset")

    save_overlap(image_files, mask_files)
