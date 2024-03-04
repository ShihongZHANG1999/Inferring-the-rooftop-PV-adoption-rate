import os
import sys
import argparse
import collections
import toml
from tqdm import tqdm
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop, Normalize
from src.transforms import (
    JointCompose,
    JointTransform,
    JointRandomHorizontalFlip,
    JointRandomRotation,
    ConvertImageMode,
    ImageToTensor,
    MaskToTensor,
)
from src.datasets import SlippyMapTilesConcatenation
from src.metrics import Metrics


def get_dataset_loaders(target_size, batch_size, dataset_path):
    target_size = (target_size, target_size)
    path = dataset_path
    # using imagenet mean and std for Normalization
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = JointCompose(
        [   
            JointTransform(ConvertImageMode("RGB"), ConvertImageMode("P")),
            JointTransform(Resize(target_size, Image.BILINEAR), Resize(target_size, Image.NEAREST)),
            JointTransform(CenterCrop(target_size), CenterCrop(target_size)),
            JointRandomHorizontalFlip(0.5),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointRandomRotation(0.5, 90),
            JointTransform(ImageToTensor(), MaskToTensor()),
            JointTransform(Normalize(mean=mean, std=std), None),
        ]
    )
    train_dataset = SlippyMapTilesConcatenation(
    os.path.join(path, "images", "training"), os.path.join(path, "annotations", "training"), transform
    )
    # print('train_dataset', train_dataset)
    val_dataset = SlippyMapTilesConcatenation(
        os.path.join(path, "images", "validation"), os.path.join(path, "annotations", "validation"), transform
    )
    assert len(train_dataset) > 0, "at least one tile in training dataset"
    assert len(val_dataset) > 0, "at least one tile in validation dataset"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def train(loader, num_classes, device, backbone, head, optimizer, optimizer_, criterion):
    num_samples = 0
    running_loss = 0

    # always two classes in our case
    metrics = Metrics(range(num_classes))
    # initialized model
    # net.train()
    backbone.train()
    head.train()
    
    # training loop
    for images, masks, _ in tqdm(loader, desc="Train", unit="batch", ascii=True):
        images = images.to(device)
        masks = masks.to(device)

        assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

        num_samples += int(images.size(0))
        optimizer.zero_grad()
        optimizer_.zero_grad()
        outputs = head(backbone(images))
        outputs = torch.nn.functional.interpolate(outputs, masks.shape[-2:], mode = 'bilinear', align_corners = True)
        # import pandas as pd
        # print(pd.Series(masks.numpy().reshape(-1)).value_counts())
        # print(outputs.shape, masks.shape)
        masks[masks!=0] = 1
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        optimizer_.step()

        running_loss += loss.item()

        for mask, output in zip(masks, outputs):
            prediction = output.detach()
            metrics.add(mask, prediction)

    return {
        "loss": running_loss / num_samples,
        "miou": metrics.get_miou(),
        "fg_iou": metrics.get_fg_iou(),
        "mcc": metrics.get_mcc(),
    }

def validate(loader, num_classes, device, backbone, head, criterion):
    num_samples = 0
    running_loss = 0

    metrics = Metrics(range(num_classes))

    with torch.no_grad():
        backbone.eval()
        head.eval()

        for images, masks, tiles in tqdm(loader, desc="Validate", unit="batch", ascii=True):
            images = images.to(device)
            masks = masks.to(device)

            assert images.size()[2:] == masks.size()[1:], "resolutions for images and masks are in sync"

            num_samples += int(images.size(0))
            outputs = head(backbone(images))
            outputs = torch.nn.functional.interpolate(outputs, masks.shape[-2:], mode = 'bilinear', align_corners = True)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            for mask, output in zip(masks, outputs):
                metrics.add(mask, output)

        return {
            "loss": running_loss / num_samples,
            "miou": metrics.get_miou(),
            "fg_iou": metrics.get_fg_iou(),
            "mcc": metrics.get_mcc(),
        }