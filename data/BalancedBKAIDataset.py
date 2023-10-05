import random
import albumentations as A

from data.batch_preprocess import *

class BalancedBKAIDataset():
    def __init__(self, config, split):
        super().__init__()
        self.base_dir = config["data_dir"]
        self.image_dir = f"{self.base_dir}/train"
        self.mask_dir = f"{self.base_dir}/train_gt"

        self.split = split
        self.size = config["img_size"]
        self.augment = config["augment"]
        self.batch_size = config["batch_size"]
        self.num_classes = config["num_classes"]
        self.mixup_alpha = config["mixup_alpha"]
        self.spatial_alpha = config["spatial_alpha"]

        self.color_dir = f"{self.base_dir}/{self.split}_colors"
        self.total_files = get_file_list(self.color_dir)

        self.file_list = []
        for name, files in self.total_files.items():
            # print(name, len(files))
            self.file_list.extend(files)

        self.batch_per_color = {"red" : config["red_size"], "green" : config["green_size"], "rng" : config["rng_size"]}
        assert self.batch_size == sum(self.batch_per_color.values()), "The batch_size does not match the total sum of batch_per_color values."

        self.train_transform = A.Compose([A.OneOf([A.CLAHE(p=0.3),
                                                   A.Sharpen(p=0.3),
                                                   A.RandomBrightnessContrast(p=0.3),], p=1), 
                                    
                                          A.OneOf([A.Rotate(limit=45, border_mode=0, p=0.25),
                                                   A.HorizontalFlip(p=0.25),
                                                   A.VerticalFlip(p=0.25),
                                                   A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.15, rotate_limit=45, border_mode=0, p=0.25),], p=1),


                                          A.OneOf([A.ColorJitter(p=0.25),
                                                   A.RandomGamma(gamma_limit=(70, 130), eps=None, always_apply=False, p=0.25),
                                                   A.ChannelDropout(p=0.25),
                                                   A.ChannelShuffle(p=0.25)], p=0.2), 


                                          A.OneOf([A.OpticalDistortion(border_mode=0, p=0.5),
                                                   A.GridDistortion(border_mode=0, p=0.5)], p=0.3),

                                    
                                          A.OneOf([A.Blur(p=0.125), 
                                                   A.GaussianBlur(p=0.125), 
                                                   A.GlassBlur(p=0.125), 
                                                   A.MotionBlur(p=0.125), 
                                                   A.GaussNoise(p=0.125), 
                                                   A.Sharpen(p=0.125), 
                                                   A.MedianBlur(p=0.125), 
                                                   A.MultiplicativeNoise(p=0.125)], p=0.4),
                                    
                                          A.OneOf([A.CoarseDropout(max_height=35, max_width=35, fill_value=0, mask_fill_value=0, p=0.5),
                                                   A.Compose([A.CropNonEmptyMaskIfExists(height=self.size-56, width=self.size-56, p=1),
                                                              
                                                              A.OneOf([A.PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=0, p=0.5),
                                                                       A.Resize(height=self.size, width=self.size, p=0.5)], p=1)
                                                              ], p=0.5)
                                                    ], p=0.4)
                                        ])

        self.paste_transform = A.Compose([A.OneOf([A.CLAHE(p=0.3),
                                                   A.Sharpen(p=0.3),
                                                   A.RandomBrightnessContrast(p=0.3),], p=1), 
                                    
                                          A.OneOf([A.HorizontalFlip(p=0.25),
                                                   A.VerticalFlip(p=0.25),], p=1),

                                    
                                          A.OneOf([A.Blur(p=0.5), 
                                                   A.GaussianBlur(p=0.5)], p=0.3)])

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
                image, mask = load_img_mask(image_file_path, mask_file_path, size=self.size)

                if self.split == "train" and self.augment:
                    prob = random.random()
                    if prob <= 0.3:
                        transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

                    elif 0.3 < prob <= 0.6:
                        piecies = [[image, mask]]
                        while len(piecies) < 4:
                            i = random.randint(0, len(color_per_files)-1)
                            file_name = color_per_files[i]
                            piece_image_path = f"{self.image_dir}/{file_name}.jpeg"
                            piece_mask_path = f"{self.mask_dir}/{file_name}.jpeg"
                            piece_image, piece_mask = load_img_mask(piece_image_path, piece_mask_path, self.size)

                            # piecies.append([piece_image, piece_mask])

                            t_piece_image, t_piece_mask = train_img_mask_transform(self.train_transform, piece_image, piece_mask)
                            piecies.append([t_piece_image, t_piece_mask])

                        transform_image, transform_mask = mosaic_augmentation(piecies, self.size)

                    elif 0.6 < prob <= 1:
                        transform_image, transform_mask = spatially_exclusive_pasting(image, mask, alpha=self.spatial_alpha)
                
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

