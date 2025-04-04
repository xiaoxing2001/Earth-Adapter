import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys
import numpy as np
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pretrained", default='sam_vit_h_4b8939.pth', type=str)
    args.add_argument("--converted", default='checkpoints/sam_vit_h_converted_512x512.pth', type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=512, type=int)
    args.add_argument("--width", default=512, type=int)
    return args.parse_args()


def select_component(d: dict, k: str):
    return {_k.replace(k, ""): v for _k, v in d.items() if k in _k}


def load_weight(pretrained_path):
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} dont exist(absolute path: {osp.abspath(pretrained_path)})"
        )
    weight = torch.load(pretrained_path, map_location="cpu")
    weight = select_component(weight, "image_encoder.")
    if len(weight.keys()) <= 10:
        print(f"The read weights may be abnormal, as shown below:")
        print(weight.keys())
        raise KeyError()
    return weight


def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16):
    assert key in weight, f"{key} must in {weight.keys()}"
    ori_shape = weight[key].shape
    weight[key] = F.interpolate(
        weight[key].float(),
        size=(kernel_conv, kernel_conv),
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = weight[key].shape
    print(f"Convert conv kernel in patch embed layer: {ori_shape} -> {dst_shape}")


def interpolate_pos_embed_(
    weight: dict, key="pos_embed", crop_size=(512, 512), kernel_conv=16
):
    pos_tokens = weight[key]
    orig_shape = pos_tokens.shape
    dst_shape = (orig_shape[0],orig_shape[-1]) + crop_size
    embed_dim = orig_shape[-1]
    # ...
    crop_size = tuple(L // kernel_conv for L in crop_size)
    resized_pos_tokens = F.interpolate(
        pos_tokens.permute(0, 3, 1, 2),
        size=crop_size,
        mode="bicubic",
        align_corners=False,
    )
    resized_pos_tokens = resized_pos_tokens.permute(0, 2, 3, 1)
    weight[key] = resized_pos_tokens
    print(
        f"Convert pos embedding: {pos_tokens.shape} -> {orig_shape} -> {dst_shape} -> {resized_pos_tokens.shape}"
    )


def main():
    args = parse_args()
    pretrained_path = args.pretrained
    converted_path = args.converted
    kernel_conv = args.kernel
    crop_size = (args.height, args.width)
    weight = load_weight(pretrained_path)
    print("Load from", pretrained_path)
    interpolate_patch_embed_(weight, kernel_conv=kernel_conv)
    interpolate_pos_embed_(weight, crop_size=crop_size, kernel_conv=kernel_conv)
    torch.save(weight, converted_path)
    print("Save to", converted_path)
    return args


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()
