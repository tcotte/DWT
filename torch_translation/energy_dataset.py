import os
from typing import Union

import cv2
import imutils
import imutils.paths
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_annotation_file_from_img_path(img_path: str, ss: bool) -> Union[str, bytes]:
    split_path = img_path.split(os.sep)

    if not ss:
        split_path[-2] = "energy_ann"
        return os.path.join(split_path[0] + "\\", *split_path[1:-1], split_path[-1].replace(".png", ".jpg"))

    else:
        split_path[-2] = "masks_machine"
        return os.path.join(split_path[0] + "\\", *split_path[1:-1], split_path[-1])



class FrenchFrieDataset(Dataset):
    def __init__(self, img_path, ann_path, transforms):
        self.img_path = img_path
        self.ann_path = ann_path
        self.transforms = transforms


    def __getitem__(self, idx):
        image_path = list(imutils.paths.list_images(self.img_path))[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        energy_annotation_file = get_annotation_file_from_img_path(img_path=image_path, ss=False)
        energy_annotation = cv2.imread(energy_annotation_file, cv2.IMREAD_GRAYSCALE)
        energy_mask = energy_annotation/255.

        ss_mask_file = get_annotation_file_from_img_path(img_path=image_path, ss=True)
        ss_mask = cv2.imread(ss_mask_file, cv2.IMREAD_GRAYSCALE)

        augmented = self.transforms(image=np.array(image), masks=np.array([ss_mask, energy_mask]))
        image = augmented['image']
        masks = augmented['masks']
        masks = torch.tensor(np.stack(masks))
        #masks = torch.tensor(np.stack(masks))
        return {'image': image.float(), 'masks': masks.float()}

    def __len__(self):
        return len(list(imutils.paths.list_images(self.img_path)))

if __name__ == '__main__':
    IMAGE_RESIZE = (1836, 2706)
    RESNET_MEAN = (0.485, 0.456, 0.406)
    RESNET_STD = (0.229, 0.224, 0.225)

    data_transforms = {
        "train": A.Compose([
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #         A.RandomRotate90(p=0.5),
            A.ColorJitter(p=0.5),
            A.ChannelShuffle(p=0.25),
            A.ToGray(p=0.25),
            #A.Normalize(RESNET_MEAN, std=RESNET_STD, p=1),
            ToTensorV2()], p=1.0),

        "valid": A.Compose([
            A.Resize(*IMAGE_RESIZE),
            A.Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1),
            ToTensorV2()], p=1.0)
    }

    ds = FrenchFrieDataset(img_path=r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\img",
                           ann_path=r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3\energy_ann",
                           transforms=data_transforms["train"])
    print(ds[0])

    plt.imshow(ds[0]["image"].numpy().transpose(1, 2, 0))
    plt.show()
    plt.imshow(ds[0]["masks"][0])
    print(np.unique(ds[0]["masks"][0]))
    plt.show()

    plt.imshow(ds[0]["masks"][1])
    print(np.unique(ds[0]["masks"][1]))
    plt.show()