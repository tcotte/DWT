import os
import shutil

import numpy as np
import segmentation_models_pytorch as smp
import imutils
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import albumentations as A
import torch.functional as F
from torch_translation.energy_dataset import get_annotation_file_from_img_path, FrenchFrieDataset
from torch_translation.utils import iou_map, watershed_energy

IMAGE_RESIZE = (512, 704)
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

MODEL_PATH = r"C:\Users\tristan_cotte\PycharmProjects\dwt\model"

data_transforms = {
    "train": A.Compose([
        A.Resize(*IMAGE_RESIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #         A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.5),
        A.ChannelShuffle(p=0.25),
        A.ToGray(p=0.25),
        # A.Normalize(RESNET_MEAN, std=RESNET_STD, p=1),
        ToTensorV2()], p=1.0),

    "valid": A.Compose([
        A.Resize(*IMAGE_RESIZE),
        A.Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1),
        ToTensorV2()], p=1.0)
}


def train_loop(model, optimizer, loader, criterion):
    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    for d in loader:
        y = d['masks'].to(device)
        pred_y = model(d['image'].to(device))
        loss = criterion(pred_y, y.float())
        losses.append(loss.item())
        step_lr = np.array([param_group["lr"] for param_group in optimizer.param_groups]).mean()
        lrs.append(step_lr)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return np.array(losses).mean(), np.array(lrs).mean()


def valid_loop(model, loader, criterion):
    losses, true_masks, pred_masks, pred_energys = [], [], [], []
    model.eval()
    for d in loader:
        with torch.no_grad():
            y = d['masks'].to(device)
            pred_y = model(d['image'].to(device))
            loss = criterion(pred_y, y.float())
        losses.append(loss.item())
        energy = torch.mean(torch.sigmoid(pred_y[:, :(mask_len - 1)]), dim=1)
        pred_masks.append(torch.sigmoid(pred_y[:, 0].cpu()))
        true_masks.append(y[:, -1].cpu())
        pred_energys.append(energy.cpu())
    pred_masks = torch.cat(pred_masks)
    true_masks = torch.cat(true_masks)
    pred_energys = torch.cat(pred_energys)
    return np.array(losses).mean(), true_masks, pred_masks, pred_energys


formed_dataset = True

if not formed_dataset:
    root_dataset = r"C:\Users\tristan_cotte\Downloads\302028_300417_KFC_Fries (#2)(1)\ds_simplest_3"
    X = list(imutils.paths.list_images(os.path.join(root_dataset, "img")))
    y = [get_annotation_file_from_img_path(i, ss=False) for i in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_ss_train = [get_annotation_file_from_img_path(i, ss=True) for i in X_train]
    y_ss_test = [get_annotation_file_from_img_path(i, ss=True) for i in X_test]

    for ds_name, ds_element in zip(['train/img', 'test/img', 'train/energy_ann', 'test/energy_ann', 'train/masks_machine', 'test/masks_machine'],
                                   [X_train, X_test, y_train, y_test, y_ss_train, y_ss_test]):
        os.makedirs(os.path.join(r"C:\Users\tristan_cotte\PycharmProjects\dwt\dataset", ds_name), exist_ok=True)

        for element in ds_element:
            shutil.copy(element, os.path.join(r"C:\Users\tristan_cotte\PycharmProjects\dwt\dataset", ds_name))

root_dataset = r'C:\Users\tristan_cotte\PycharmProjects\dwt\dataset'


# X_train = list(imutils.paths.list_images(os.path.join(root_dataset, "train", "img")))
# y_train = [get_annotation_file_from_img_path(i) for i in X_train]
# X_test = list(imutils.paths.list_images(os.path.join(root_dataset, "test", "img")))
# y_test = [get_annotation_file_from_img_path(i) for i in X_test]
train_dataset = FrenchFrieDataset(img_path=os.path.join(root_dataset, "train", "img"),
                                  ann_path=os.path.join(root_dataset, "train", "energy_ann"),
                                  transforms=data_transforms["train"])

test_dataset = FrenchFrieDataset(img_path=os.path.join(root_dataset, "test", "img"),
                                 ann_path=os.path.join(root_dataset, "test", "energy_ann"),
                                 transforms=data_transforms["valid"])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

model_name = "resnet50"
mask_len = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 5e-4
min_lr = 5e-5
n_epochs = 50

model = smp.Unet(model_name, encoder_weights="imagenet", activation=None, classes=mask_len)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = smp.losses.JaccardLoss(mode='multilabel')
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)

valid_best_score = 0.

for epoch in tqdm(range(n_epochs)):
    train_loss, lrs = train_loop(model, optimizer, train_loader, criterion)
    valid_loss, valid_mask, valid_pred_mask, valid_pred_energy = valid_loop(model, test_loader, criterion)

    preds_wt = torch.stack(
        [torch.tensor(watershed_energy(pred.numpy() * 255, energy.numpy() * 255, 0.5, 0.7)) for pred, energy in
         zip(valid_pred_mask, valid_pred_energy)])
    valid_score_energy = iou_map(valid_mask, preds_wt)
    if valid_score_energy > valid_best_score:
        print(
            f"epoch: {epoch}, train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}, meanIoU: {valid_score_energy:.3f}")
        valid_best_score = valid_score_energy
        torch.save(model.state_dict(), f'{MODEL_PATH}/{model_name}.pth')
    scheduler.step()
