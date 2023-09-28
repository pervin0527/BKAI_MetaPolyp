import random
import albumentations as A

from data.batch_preprocess import *

class BalancedBKAIDataset():
    def __init__(self, config, split):
        super().__init__()
        self.base_dir = config["data_dir"]
        self.image_dir = f"{self.base_dir}/train"
        self.mask_dir = f"{self.base_dir}/train_mask"

        self.split = split
        self.alpha = config["alpha"]
        self.size = config["img_size"]
        self.augment = config["augment"]
        self.batch_size = config["batch_size"]
        self.num_classes = config["num_classes"]

        self.color_dir = f"{self.base_dir}/{self.split}_colors"
        self.total_files = get_file_list(self.color_dir)

        self.file_list = []
        for name, files in self.total_files.items():
            # print(name, len(files))
            self.file_list.extend(files)

        self.batch_per_color = {"red" : config["red_size"], "green" : config["green_size"], "rng" : config["rng_size"]}
        assert self.batch_size == sum(self.batch_per_color.values()), "The batch_size does not match the total sum of batch_per_color values."

        self.train_transform = A.Compose([A.Rotate(limit=90, border_mode=0, p=0.6),
                                          A.HorizontalFlip(p=0.7),
                                          A.VerticalFlip(p=0.7),
                                          
                                          A.ShiftScaleRotate(shift_limit_x=(-0.06, 0.06), shift_limit_y=(-0.06, 0.06), 
                                                             scale_limit=(-0.25, 0.25), 
                                                             rotate_limit=(-90, 90), 
                                                             interpolation=0, border_mode=0, 
                                                             value=(0, 0, 0), mask_value=None, rotate_method='largest_box',
                                                             p=0.5),
                                          A.GridDistortion(num_steps=5, distort_limit=(-0.3, 0.3), 
                                                           interpolation=0, border_mode=0, 
                                                           value=(0, 0, 0), mask_value=None, normalized=False,
                                                           p=0.5)])   
        
        self.image_transform = A.Compose([A.Blur(p=0.4),
                                          A.RandomBrightnessContrast(p=0.8),
                                          A.CLAHE(p=0.5)])

    def __len__(self):
        return len(self.file_list)
        

    def __getitem__(self, index):
        batch_images, batch_masks = [], []
        for color, batch_size in self.batch_per_color.items():
            color_per_files = self.total_files[color]
            random.shuffle(color_per_files)

            for idx in range(batch_size):
                file_name = color_per_files[idx]
                image_file_path = f"{self.image_dir}/{file_name}.jpeg"
                mask_file_path = f"{self.mask_dir}/{file_name}.jpeg"
                image, mask = load_img_mask(image_file_path, mask_file_path, self.size)

                if self.split == "train" and self.augment:
                    prob = random.random()
                    if prob <= 0.25:
                        transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

                    elif 0.25 < prob <= 0.5:
                        piecies = [[image, mask]]
                        while len(piecies) < 4:
                            i = random.randint(0, len(color_per_files)-1)
                            file_name = color_per_files[i]
                            piece_image = f"{self.image_dir}/{file_name}.jpeg"
                            piece_mask = f"{self.mask_dir}/{file_name}.jpeg"

                            piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)
                            t_piece_image, t_piece_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)
                            piecies.append([t_piece_image, t_piece_mask])

                        transform_image, transform_mask = mosaic_augmentation(piecies, self.size)

                    elif 0.5 < prob <= 0.65:
                        i = random.randint(0, len(color_per_files)-1)
                        file_name = color_per_files[i]
                        piece_image = f"{self.image_dir}/{file_name}.jpeg"
                        piece_mask = f"{self.mask_dir}/{file_name}.jpeg"
                        piece_image, piece_mask = load_img_mask(piece_image, piece_mask, self.size)

                        t_piece_image, t_piece_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)
                        transform_image, transform_mask = cutmix_augmentation(image, mask, t_piece_image, t_piece_mask)

                    elif 0.65 < prob <= 1:
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

