# Modified from https://github.com/PruneTruong/DenseMatching
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import cv2
import numpy as np

def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    if torch.all(flo == 0):
        if return_mask:
            return x, torch.ones((B, H, W), dtype=torch.bool, device=x.device)
        return x
    # mesh grid
    xx = torch.arange(0, W, dtype=flo.dtype,
                      device=flo.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, dtype=flo.dtype,
                      device=flo.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / float(max(W-1, 1)) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / float(max(H-1, 1)) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    # force full prec for AMP (fixed in 1.10)
    with autocast(enabled=False):
        output = nn.functional.grid_sample(
            x.float(), vgrid.float(), align_corners=True, padding_mode=padding_mode)
    if return_mask:
        vgrid = vgrid.detach().clone().permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -
                                     1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output

def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    Opencv remap
    map_x contains the index of the matching horizontal position of each pixel [i,j] while map_y contains the
    index of the matching vertical position of each pixel [i,j]
    All arrays are numpy
    args:
        image: image to remap, HxWxC
        disp_x: displacement in the horizontal direction to apply to each pixel. must be float32. HxW
        disp_y: displacement in the vertical direction to apply to each pixel. must be float32. HxW
        interpolation
        border_mode
    output:
        remapped image. HxWxC
    """
    h_scale, w_scale=disp_x.shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return remapped_image

def estimate_probability_of_confidence_interval_of_mixture_density(uncert_output, R=1.0):
    # NOTE: ONLY FOR GAUSSIAN
    assert uncert_output.shape[1] == 1
    var = torch.exp(uncert_output)
    p_r = 1.0 - torch.exp(-R ** 2 / (2 * var))
    return p_r


def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if len(mapping.shape) == 4:
        # shape is B,C,H,W
        b, _, h, w = mapping.shape
        mask = mapping[:, 0].ge(0) & mapping[:, 0].le(
            w - 1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h - 1)
    else:
        _, h, w = mapping.shape
        mask = mapping[0].ge(0) & mapping[0].le(
            w - 1) & mapping[1].ge(0) & mapping[1].le(h - 1)
    mask = mask.bool()
    return mask


def unnormalise_and_convert_mapping_to_flow(map, output_channel_first=True):
    if len(map.shape) == 4:
        if map.shape[1] != 2:
            # load_size is BxHxWx2
            map = map.permute(0, 3, 1, 2)

        # channel first, here map is normalised to -1;1
        # we put it back to 0,W-1, then convert it to flow
        B, C, H, W = map.size()
        mapping = torch.zeros_like(map)
        # mesh grid
        mapping[:, 0, :, :] = (map[:, 0, :, :] + 1) * \
            (W - 1) / 2.0  # unormalise
        mapping[:, 1, :, :] = (map[:, 1, :, :] + 1) * \
            (H - 1) / 2.0  # unormalise

        xx = torch.arange(0, W, dtype=mapping.dtype,
                          device=mapping.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=mapping.dtype,
                          device=mapping.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)

        flow = mapping - grid  # here also channel first
        if not output_channel_first:
            flow = flow.permute(0, 2, 3, 1)
    else:
        if map.shape[0] != 2:
            # load_size is HxWx2
            map = map.permute(2, 0, 1)

        # channel first, here map is normalised to -1;1
        # we put it back to 0,W-1, then convert it to flow
        C, H, W = map.size()
        mapping = torch.zeros_like(map)
        # mesh grid
        mapping[0, :, :] = (map[0, :, :] + 1) * (W - 1) / 2.0  # unormalise
        mapping[1, :, :] = (map[1, :, :] + 1) * (H - 1) / 2.0  # unormalise

        xx = torch.arange(0, W, dtype=mapping.dtype,
                          device=mapping.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=mapping.dtype,
                          device=mapping.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0)  # attention, concat axis=0 here

        flow = mapping - grid  # here also channel first
        if not output_channel_first:
            flow = flow.permute(1, 2, 0)
    return flow


def create_border_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if len(mapping.shape) == 4:
        # shape is B,C,H,W
        b, _, h, w = mapping.shape
        mask = mapping[:, 0].ge(0) & mapping[:, 0].le(
            w - 1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h - 1)
    else:
        _, h, w = mapping.shape
        mask = mapping[0].ge(0) & mapping[0].le(
            w - 1) & mapping[1].ge(0) & mapping[1].le(h - 1)
    mask = mask.bool()
    return mask


def convert_flow_to_mapping(flow, output_channel_first=True):
    if len(flow.shape) == 4:
        if flow.shape[1] != 2:
            # load_size is BxHxWx2
            flow = flow.permute(0, 3, 1, 2)

        B, C, H, W = flow.size()

        xx = torch.arange(0, W, dtype=flow.dtype,
                          device=flow.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=flow.dtype,
                          device=flow.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)

        mapping = flow + grid  # here also channel first
        if not output_channel_first:
            mapping = mapping.permute(0, 2, 3, 1)
    else:
        if flow.shape[0] != 2:
            # load_size is HxWx2
            flow = flow.permute(2, 0, 1)

        C, H, W = flow.size()

        xx = torch.arange(0, W, dtype=flow.dtype,
                          device=flow.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=flow.dtype,
                          device=flow.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0)  # attention, concat axis=0 here

        mapping = flow + grid  # here also channel first
        if not output_channel_first:
            mapping = mapping.permute(1, 2, 0)
    return mapping


def convert_mapping_to_flow(mapping, output_channel_first=True):
    if len(mapping.shape) == 4:
        if mapping.shape[1] != 2:
            # load_size is BxHxWx2
            mapping = mapping.permute(0, 3, 1, 2)

        B, C, H, W = mapping.size()

        xx = torch.arange(0, W, dtype=mapping.dtype,
                          device=mapping.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=mapping.dtype,
                          device=mapping.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1)

        flow = mapping - grid  # here also channel first
        if not output_channel_first:
            flow = flow.permute(0, 2, 3, 1)
    else:
        if mapping.shape[0] != 2:
            # load_size is HxWx2
            mapping = mapping.permute(2, 0, 1)

        C, H, W = mapping.size()

        xx = torch.arange(0, W, dtype=mapping.dtype,
                          device=mapping.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, dtype=mapping.dtype,
                          device=mapping.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0)  # attention, concat axis=0 here

        flow = mapping - grid  # here also channel first
        if not output_channel_first:
            flow = flow.permute(1, 2, 0)
    return flow
