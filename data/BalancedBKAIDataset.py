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

        self.train_transform = A.Compose([A.CLAHE(p=0.4),
                                          A.RandomBrightnessContrast(p=0.4),
                                          A.Rotate(limit=90, border_mode=0, p=0.6),
                                          A.HorizontalFlip(p=0.5),
                                          A.VerticalFlip(p=0.5),
                                          A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
                                          A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                                          A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
                                          A.CoarseDropout(p=0.2, max_height=35, max_width=35, fill_value=255, mask_fill_value=0),
                                          A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15)])

        crop_size = config["img_size"] - 76
        self.letter_box_transform = A.Compose([
            A.RandomResizedCrop(height=crop_size, width=crop_size, p=1),
            A.PadIfNeeded(p=1.0, min_height=config["img_size"], min_width=config["img_size"], pad_height_divisor=None, pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None)])

        self.piece_transform = A.Compose([
            A.CLAHE(p=0.4),
            A.RandomBrightnessContrast(p=0.4),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)])

        background_dir = config["background_dir"]
        sets = ["train", "val", "test"]
        targets = ["0_normal", "1_ulcerative_colitis", "3_esophagitis"]

        self.total_bg_files = []
        for ds in sets:
            for target in targets:
                files = sorted(glob(f"{background_dir}/{ds}/{target}/*"))
                self.total_bg_files.extend(files)

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
                    if prob <= 0.2:
                        transform_image, transform_mask = train_img_mask_transform(self.train_transform, image, mask)

                        if random.random() > 0.5:
                            transform_image, transform_mask = train_img_mask_transform(self.letter_box_transform, transform_image, transform_mask)

                    elif 0.2 < prob <= 0.5:
                        piecies = [[image, mask]]
                        while len(piecies) < 4:
                            i = random.randint(0, len(color_per_files)-1)
                            file_name = color_per_files[i]
                            piece_image_path = f"{self.image_dir}/{file_name}.jpeg"
                            piece_mask_path = f"{self.mask_dir}/{file_name}.jpeg"
                            piece_image, piece_mask = load_img_mask(piece_image_path, piece_mask_path, self.size)

                            # piecies.append([piece_image, piece_mask])

                            t_piece_image, t_piece_mask = train_img_mask_transform(self.piece_transform, piece_image, piece_mask)
                            piecies.append([t_piece_image, t_piece_mask])

                        transform_image, transform_mask = mosaic_augmentation(piecies, self.size)

                    elif 0.5 < prob <= 8:
                        t_image, t_mask = train_img_mask_transform(self.piece_transform, image, mask)
                        transform_image, transform_mask = spatially_exclusive_pasting(image=t_image, mask=t_mask, alpha=self.spatial_alpha)
                        # transform_image, transform_mask = spatially_exclusive_pasting(image, mask, alpha=self.alpha)

                    elif 0.8 < prob < 1:
                        bg_idx = random.randint(0, len(self.total_bg_files) - 1)
                        bg_file = self.total_bg_files[bg_idx]
                        bg_image = load_img_mask(image_path=bg_file, mask_path=None, size=self.size, only_img=True)
                        transform_image, transform_mask = background_pasting(image, mask, bg_image, alpha=self.spatial_alpha)
                
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

