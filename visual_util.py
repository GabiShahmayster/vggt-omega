# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import numpy as np
import requests


def filter_points(
    predictions: dict,
    conf_thres: float = 20.0,
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    image_dir: str | None = None,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
    depth_edge_rtol: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter VGGT-Omega camera/depth predictions down to a colored point cloud."""
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    conf_thres = max(2.0, float(conf_thres))

    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"]
    if filter_depth_edges and "depth" in predictions:
        conf = conf.copy()
        conf[depth_edge(predictions["depth"][..., 0], rtol=depth_edge_rtol)] = 0.0
    images = predictions["images"]

    if mask_sky and image_dir is not None:
        conf = apply_sky_mask(conf, image_dir)

    vertices = points.reshape(-1, 3)
    colors = _images_to_rgb(images).reshape(-1, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_thres > 0 and np.any(mask):
        conf_threshold = np.percentile(conf[mask], conf_thres)
        mask &= conf >= conf_threshold
    mask &= conf > 1e-5

    if mask_black_bg:
        mask &= colors.sum(axis=1) >= 16
    if mask_white_bg:
        mask &= ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))

    vertices = vertices[mask]
    colors = colors[mask]
    return _limit_points(vertices, colors, max_points)


def _images_to_rgb(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def _limit_points(vertices: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors
    indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[indices], colors[indices]


def depth_edge(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def apply_sky_mask(conf: np.ndarray, image_dir: str) -> np.ndarray:
    image_names = sorted(os.listdir(image_dir))
    height, width = conf.shape[-2:]
    masks = []
    skyseg_session = None

    for image_name in image_names:
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists("skyseg.onnx"):
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                "skyseg.onnx",
            )
        if skyseg_session is None:
            import onnxruntime

            skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
        sky_mask = segment_sky(image_path, skyseg_session)

        if sky_mask.shape != (height, width):
            sky_mask = cv2.resize(sky_mask, (width, height))
        masks.append(sky_mask)

    return conf * (np.array(masks) > 0.1).astype(np.float32)


def segment_sky(image_path: str, onnx_session) -> np.ndarray:
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, [320, 320], image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map)
    output_mask[result_map < 32] = 255
    return output_mask


def run_skyseg(onnx_session, input_size: list[int], image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image = (image / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose(2, 0, 1)
    image = image.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    result = onnx_session.run([output_name], {input_name: image})
    result = np.array(result).squeeze()
    result_min = np.min(result)
    result_max = np.max(result)
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)
    else:
        result = np.zeros_like(result)
    return (result * 255).astype("uint8")


def download_file_from_url(url: str, filename: str) -> None:
    tmp_filename = f"{filename}.tmp"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(tmp_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    os.replace(tmp_filename, filename)
