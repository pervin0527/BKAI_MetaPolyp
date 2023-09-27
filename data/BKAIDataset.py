import random
import albumentations as A

from data.batch_preprocess import *

class BKAIDataset():
    def __init__(self, config, split):
        super().__init__()
        self.base_dir = config["data_dir"]
        self.image_dir = f"{self.base_dir}/train"
        self.mask_dir = f"{self.base_dir}/train_mask"

        self.split = split
        self.alpha = config["alpha"]
        self.size = config["img_size"]
        self.augment = config["augment"]
        self.times = config["mixup_times"]
        self.batch_size = config["batch_size"]
        self.num_classes = config["num_classes"]

        self.train_transform = A.Compose([A.Rotate(limit=90, border_mode=0, p=0.6),
                                          A.HorizontalFlip(p=0.7),
                                          A.VerticalFlip(p=0.7)])   
        
        self.image_transform = A.Compose([A.Blur(p=0.4),
                                          A.RandomBrightnessContrast(p=0.8),
                                          A.CLAHE(p=0.5)])

        self.split_txt = f"{self.base_dir}/{split}.txt"
        with open(self.split_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]   


    def __len__(self):
        return len(self.file_list)
        

    def __getitem__(self, index):
        batch_images, batch_masks = [], []
        for batch_idx in range(self.batch_size):
            bi = (index * self.batch_size + batch_idx) % len(self.file_list)
            file_name = self.file_list[bi]
            
            image_file_path = f"{self.image_dir}/{file_name}.jpeg"
            mask_file_path = f"{self.mask_dir}/{file_name}.jpeg"
            image, mask = load_img_mask(image_file_path, mask_file_path, self.size)

            if self.split == "train" and self.augment:
                prob = random.random()
                if prob <= 25:
                    transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

                elif 0.25 < prob <= 0.5:
                    piecies = [[image, mask]]
                    while len(piecies) < 4:
                        i = random.randint(0, len(self.file_list)-1)
                        file_name = self.file_list[i]
                        piece_image = f"{self.image_dir}/{file_name}.jpeg"
                        piece_mask = f"{self.mask_dir}/{file_name}.jpeg"

                        piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)
                        transform_image, transform_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)
                        piecies.append([transform_image, transform_mask])

                    transform_image, transform_mask = mosaic_augmentation(piecies, self.size)

                elif 0.5 < prob <= 0.75:
                    if random.random() > 0.2:
                        i = random.randint(0, len(self.file_list)-1)
                        file_name = self.file_list[i]
                        piece_image = f"{self.image_dir}/{file_name}.jpeg"
                        piece_mask = f"{self.mask_dir}/{file_name}.jpeg"
                    else:
                        piece_image, piece_mask = np.zeros_like(mask), np.zeros_like(mask)

                    piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)
                    transform_image, transform_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)

                    transform_image, transform_mask = cutmix_augmentation(image, mask, piece_image, piece_mask)

                elif 0.75 < prob <= 1:
                    transform_image, transform_mask = spatially_exclusive_pasting(image, mask, alpha=self.alpha)
            
                batch_image = normalize(transform_image)
                batch_mask = encode_mask(transform_mask)

                nc = self.num_classes
                one_hot_batch_mask = np.eye(nc)[batch_mask.astype(int)]

                batch_images.append(batch_image)
                batch_masks.append(one_hot_batch_mask)
            
            else:
                batch_image = normalize(image)
                batch_mask = encode_mask(mask)

                nc = self.num_classes
                one_hot_batch_mask = np.eye(nc)[batch_mask.astype(int)]

                batch_images.append(batch_image)
                batch_masks.append(one_hot_batch_mask)

        return np.stack(batch_images), np.stack(batch_masks)