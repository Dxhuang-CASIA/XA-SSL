import sys, types
import torch

import numpy.random
import os
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unext.model import UNext_S
from tqdm import tqdm
from dataset import VesselDataset
import torchvision.transforms as transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class BCEDiceLoss:
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        self.bce_weight = bce_weight
        self.smooth = smooth

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy(pred, target)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)


def train_model(
        model,
        train_loader,
        device,
        epochs,
        lr,
        ckpt,
        bce_weight,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / epochs) ** 0.9)
    criterion = BCEDiceLoss(bce_weight=bce_weight)

    loss_list = []

    for epoch in range(epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=160)
        for step, (img_tensor, mask_tensor, _) in loop:
            image = img_tensor.to(device)
            mask = mask_tensor.to(device)

            optimizer.zero_grad()
            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            loop.set_description('TRAIN ({}) | Loss: {:.4f} |'.format(epoch + 1, np.mean(loss_list)))

        lr_scheduler.step()
        loss_list = []

        if (epoch + 1) % epochs == 0:
            path = f'unext_epoch{epoch + 1}_(filter80).pth'
            torch.save(model.state_dict(), os.path.join(ckpt, path))


if __name__ == "__main__":
    seed = 30
    batch_size = 256
    epochs = 200
    lr = 1e-3
    bce_weight = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = fr"/path/to/save/checkpoints"
    set_seed(seed)

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = VesselDataset(r"/path/to/dataset")

    train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(seed))

    model = UNext_S(num_classes=1, img_size=224).to(device)

    train_model(
        model,
        train_loader,
        device,
        epochs,
        lr,
        ckpt,
        bce_weight
        )