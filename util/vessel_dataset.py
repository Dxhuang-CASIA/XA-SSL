import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import (
    InterpolationMode,
    Normalize,
    ToTensor
)

def create_image_mask_path(base_path, dataset_name):
    image_path = os.path.join(base_path, dataset_name)
    mask_path = os.path.join(base_path, dataset_name + "_filter_92")
    item_list = os.listdir(image_path)
    image_item = [os.path.join(image_path, i) for i in item_list]
    mask_item = [os.path.join(mask_path, i) for i in item_list]
    return image_item, mask_item


class VesselDataset(Dataset):
    def __init__(self, base_path, crop_size=224, scale=(0.2, 1.0),
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        
        coronarydominance_images, coronarydominance_masks= create_image_mask_path(base_path, "coronarydominance")
        xcad_images, xcad_masks = create_image_mask_path(base_path, "xcad")
        cadica_images, cadica_masks = create_image_mask_path(base_path, "cadica")
        syntax_images, syntax_masks = create_image_mask_path(base_path, "syntax")

        self.image_list = coronarydominance_images + xcad_images + cadica_images + syntax_images
        self.mask_list  = coronarydominance_masks + xcad_masks + cadica_masks + syntax_masks

        assert len(self.image_list) == len(self.mask_list), "Error: Image and mask counts do not match."

        self.crop_size = crop_size
        self.scale = scale
        self.ratio = (3/4, 4/3)
        self.normalize = Normalize(mean=mean, std=std)
        self.to_tensor = ToTensor()

        print('Length of XA-170k dataset is %d' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        msk_path = self.mask_list[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(msk_path).convert("L")

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )
        image = F.resized_crop(image, i, j, h, w, size=(self.crop_size, self.crop_size), interpolation=InterpolationMode.BICUBIC)
        mask = F.resized_crop(mask, i, j, h, w, size=(self.crop_size, self.crop_size), interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image = F.hflip(image)
            mask  = F.hflip(mask)
        
        image = self.to_tensor(image)
        image = self.normalize(image)
        mask  = self.to_tensor(mask)

        return image, mask.float(), 0