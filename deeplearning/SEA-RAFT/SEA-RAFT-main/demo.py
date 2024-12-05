import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

#from config.parser import parse_args

import datasets
from core.raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt

global device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=10)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=0, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=4, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    scale = -1
    img1 = F.interpolate(image1, scale_factor=2 ** scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** scale, mode='bilinear', align_corners=False) * (0.5 ** -1)
    info_down = F.interpolate(info, scale_factor=0.5 ** scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt):
    path = f"./custom"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"./custom/image1prueba.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"./custom/image2prueba.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)
    diff = flow_gt - flow
    diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}error_final.jpg", diff_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    print(f"EPE: {epe.mean().cpu().item()}")

@torch.no_grad()
def demo_chairs(model, args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = datasets.FlyingChairs(split='training')
    image1, image2, flow_gt, _ = dataset[1345]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('chairs', args, model, image1, image2, flow_gt)

def demo_sintel(model, args):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dstype = 'final'
    dataset = datasets.MpiSintel(split='training', dstype=dstype)
    image1, image2, flow_gt, _ = dataset[100]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_spring(model, args, split='train'):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = datasets.SpringFlowDataset(split=split)
    idx = 19198
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[idx]
    else:
        image1, image2,  _ = dataset[idx]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_tartanair(model, args, device=device):
    dataset = datasets.TartanAir()
    image1, image2, flow_gt, _ = dataset[1070]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('tartanair', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_infinigen(model, args, device=device):
    dataset = datasets.Infinigen()
    image1, image2, flow_gt, _ = dataset[1000]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('infinigen', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_hd1k(model, args, device=device):
    dataset = datasets.HD1K()
    print(len(dataset))
    image1, image2, flow_gt, _ = dataset[0]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('hd1k', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_middlebury(model, args, device=device):
    dataset = datasets.Middlebury()
    image1, image2, flow_gt, _ = dataset[3]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('middlebury', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_custom(model, args, device=device):
    image1 = cv2.imread('C:\\Users\\Susana\\Desktop\\Universidad\\Cuarto\\PIAV\\Pract\\P07\\deeplearning\\SEA-RAFT\\SEA-RAFT-main\\custom\\image1.jpg')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread('C:\\Users\\Susana\\Desktop\\Universidad\\Cuarto\\PIAV\\Pract\\P07\\deeplearning\\SEA-RAFT\\SEA-RAFT-main\\custom\\image2.jpg')
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    flow_gt = torch.zeros([2, H, W], device=device)
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('custom_downsample', args, model, image1, image2, flow_gt)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='checkpoint path', required=True, type=str)

    args = parser.parse_args([
    '--cfg', 'C:\\Users\\Susana\\Desktop\\Universidad\\Cuarto\\PIAV\\Pract\\P07\\deeplearning\\SEA-RAFT\\SEA-RAFT-main\\config\\eval\\spring-M.json',
    '--model', 'C:\\Users\\Susana\\Desktop\\Universidad\\Cuarto\\PIAV\\Pract\\P07\\deeplearning\\models\\Tartan-C-T-TSKH-spring540x960-M.pth'
    ])


    return args

def main():

    args = parse_args()
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.to(device)
    model.eval()

    demo_custom(model, args)


if __name__ == '__main__':
    main()