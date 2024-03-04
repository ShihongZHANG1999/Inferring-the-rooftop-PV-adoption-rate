import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm
from PIL import Image
from network import *
from src.datasets import BufferedSlippyMapDirectory
from src.transforms import ConvertImageMode, ImageToTensor
from src.colors import make_palette
from src.metrics import Metrics

def predict(input_dir, output_dir, image_size, device, batch_size, chkpt_backbone, chkpt_head):

    num_classes = 2
    backbone = MixVisionTransformer(model_type='mit_b5', pretrained='imagenet')
    head = DAFormerHead(in_channels=[64, 128, 320, 512], in_index=[0, 1, 2, 3], num_classes=2,
                        input_transform='multiple_select')
    backbone = backbone.to(device)
    head = head.to(device)
    backbone.load_state_dict(chkpt_backbone["state_dict"])
    head.load_state_dict(chkpt_head["state_dict"])
    backbone.eval()
    head.eval()
    metrics = Metrics(range(num_classes))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])
    directory = BufferedSlippyMapDirectory(input_dir, input_dir, transform=transform, size=image_size)
    assert len(directory) > 0, "at least one tile in dataset"

    loader = DataLoader(directory, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", unit="batch", ascii=True):
            images, labels, name = batch
            images = images.to(device)
            labels = labels.to(device)  # Tensor:(64,256,256)
            outputs = head(backbone(images))  # Tensor:(64,2,256,256)
            outputs = torch.nn.functional.interpolate(outputs, (256, 256), mode='bilinear', align_corners=True)
            labels[labels != 0] = 1  # Tensor:(64,256,256)
            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()  # Tensor:(64,2,256,256)

            for prob, output, label, output_name in zip(probs, outputs, labels, name):

                metrics.add(label, output)
                mask = np.argmax(prob, axis=0)
                mask = mask * 200
                mask = mask.astype(np.uint8)
                palette = make_palette("dark", "light")
                out = Image.fromarray(mask, mode="P")
                out.putpalette(palette)

                save_path = output_dir + '/' + output_name.split('.')[0].split('_')[0]
                if not os.path.exists(save_path):
                    os.makedirs(os.path.join(save_path), exist_ok=True)
                path = os.path.join(save_path, output_name.split('.')[0].split('_')[1] + '.png')
                out.save(path, optimize=True)

if __name__ == '__main__':
    input_dir = './prediction_demo/input'
    output_dir = './prediction_demo/output'
    image_size = 256
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    checkpoint_backbone = './checkpoint/Solar-checkpoint_backbone-1000-of-10000.pth'
    checkpoint_head = './checkpoint/Solar-checkpoint_head-1000-of-10000.pth'

    chkpt_backbone = torch.load(checkpoint_backbone, map_location=device)
    chkpt_head = torch.load(checkpoint_head, map_location=device)

    predict(input_dir, output_dir, image_size, device, batch_size, chkpt_backbone, chkpt_head)
