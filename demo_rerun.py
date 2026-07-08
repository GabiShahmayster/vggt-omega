# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import numpy as np
import rerun as rr
import torch

from visual_util import filter_points
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def run_model(image_folder: str, model: VGGTOmega, image_resolution: int) -> dict:
    print(f"Processing images from {image_folder}")

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    if len(image_names) == 0:
        raise FileNotFoundError(f"No images found in {image_folder}")

    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )

    torch.cuda.empty_cache()
    return predictions_np


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def log_to_rerun(
    predictions: dict,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    image_folder: str,
    max_points: int,
) -> None:
    vertices, colors = filter_points(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        image_dir=image_folder,
        max_points=max_points,
    )
    rr.log("world/points", rr.Points3D(positions=vertices, colors=colors))

    if not show_cam:
        return

    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    for i in range(len(extrinsic)):
        world_to_camera = np.eye(4)
        world_to_camera[:3, :4] = extrinsic[i]
        camera_to_world = np.linalg.inv(world_to_camera)

        rr.log(
            f"world/camera_{i}",
            rr.Transform3D(translation=camera_to_world[:3, 3], mat3x3=camera_to_world[:3, :3]),
        )
        rr.log(
            f"world/camera_{i}/image",
            rr.Pinhole(
                image_from_camera=intrinsic[i],
                width=images.shape[2],
                height=images.shape[1],
            ),
        )
        rr.log(f"world/camera_{i}/image", rr.Image((images[i] * 255).clip(0, 255).astype(np.uint8)))


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega rerun visualization script")
    parser.add_argument("image_folder", help="Path to a folder of images to reconstruct.")
    parser.add_argument("--checkpoint", default="../vggt-omega/vggt_omega_1b_512.pt", help="Local VGGT-Omega checkpoint path.")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution. Default: 512.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Confidence threshold percentile (0-100). Default: 50.")
    parser.add_argument("--max-points-k", type=int, default=1000, help="Max points to display, in thousands. Default: 1000.")
    parser.add_argument("--show-cam", action=argparse.BooleanOptionalAction, default=True, help="Show camera frustums.")
    parser.add_argument("--mask-sky", action=argparse.BooleanOptionalAction, default=False, help="Filter sky points.")
    parser.add_argument("--mask-black-bg", action=argparse.BooleanOptionalAction, default=False, help="Filter black background points.")
    parser.add_argument("--mask-white-bg", action=argparse.BooleanOptionalAction, default=False, help="Filter white background points.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint)
    predictions = run_model(args.image_folder, model, args.image_resolution)

    rr.init("vggt-omega", spawn=True)
    log_to_rerun(
        predictions,
        conf_thres=args.conf_thres,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        max_points=int(args.max_points_k * 1000),
    )


if __name__ == "__main__":
    main()
