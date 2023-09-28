import os
import random

def split_files(ratios):
    train_dir = f"{dir}/train_colors"
    valid_dir = f"{dir}/valid_colors"
    if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

    for name, number in ratios.items():
        with open(f"{txt_dir}/{name}.txt", "r") as f:
            files = [x.strip() for x in f.readlines()]
        
        train_files = files[:number]
        valid_files = files[number:]

        with open(f"{train_dir}/{name}.txt", "w") as tf:
            for idx, file in enumerate(train_files):
                tf.write(file)

                if idx < len(train_files):
                    tf.write("\n")

        with open(f"{valid_dir}/{name}.txt", "w") as vf:
            for idx, file in enumerate(valid_files):
                vf.write(file)

                if idx < len(valid_files):
                    vf.write("\n")



if __name__ == "__main__":
    dir = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    txt_dir = f"{dir}/color_txt"

    ratios = {"red" : 554, "green" : 207, "rng" : 39}
    split_files(ratios)