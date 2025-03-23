#!/usr/bin/env python3

"""Infers pose from objects."""

import datetime

import os
import gc
import time

from typing import List, NamedTuple, Optional, Tuple

from PIL import Image
from utils import (
    projector_util,
    repre_util,
    vis_base_util,
    renderer_base,
    render_vis_util,
    logging, 
    structs, 
    misc, 
    geometry
)

import cv2

import numpy as np

import torch

from utils.misc import array_to_tensor, tensor_to_array, tensors_to_arrays

from bop_toolkit_lib import inout, dataset_params
import bop_toolkit_lib.config as bop_config
import bop_toolkit_lib.misc as bop_misc


from utils import (
    corresp_util,
    config_util,
    eval_errors,
    eval_util,
    feature_util,
    infer_pose_util,
    knn_util,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    vis_util,
    data_util,
    renderer_builder,
    json_util, 
    logging,
    misc,
    structs,
)

from utils.structs import AlignedBox2f, PinholePlaneCameraModel
from utils.misc import warp_depth_image, warp_image

import torch.nn.functional as F
from sklearn.decomposition import PCA


logger: logging.Logger = logging.get_logger()


class InferOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    repre_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    max_sym_disc_step: float = 0.01

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Object instance options.
    use_detections: bool = True
    num_preds_factor: float = 1.0
    min_visibility: float = 0.1

    # Feature extraction options.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True

    final_pose_type: str = "best_coarse"

    # Other options.
    save_estimates: bool = True
    vis_results: bool = True
    vis_corresp_top_n: int = 100
    vis_feat_map: bool = True
    vis_for_paper: bool = True
    debug: bool = True


def infer(opts: InferOpts) -> None:

    datasets_path = bop_config.datasets_path

    # Prepare feature extractor.
    print("The opts.extractor_name is: ", opts.extractor_name)
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    # Generate grid points at which to sample the feature vectors.
    grid_size = opts.crop_size
    grid_points = feature_util.generate_grid_points(
        grid_size=grid_size,
        cell_size=opts.grid_cell_size,
    )
    grid_points = grid_points.to(device)

    # 0392, 0434
    image = np.array(Image.open("bop_datasets/templates/v1/lmo/1/rgb/template_0392.png"))
    print(image.shape, image.dtype)

    image_np_hwc = image.astype(np.float32)/255.0

    # Extract feature map.
    image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
    image_tensor_bchw = image_tensor_chw.unsqueeze(0)
    
    # Pass the image through the extractor
    extractor_output = extractor(image_tensor_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]

    # Resize feature map to match input image size using nearest neighbor interpolation
    image_height, image_width = opts.crop_size
    feature_map_chw_up = F.interpolate(
        feature_map_chw.unsqueeze(0),
        size=(image_height, image_width),
        mode="nearest"
    )[0]

    # Convert to HWC format
    feature_map_hwc = tensor_to_array(feature_map_chw_up.permute(1, 2, 0))  # HWC

    # Flatten spatial dimensions for PCA
    _c = feature_map_hwc.shape[2]
    feature_map_flat = feature_map_hwc.reshape(-1, _c)

    # Apply PCA to reduce channels to 3
    pca = PCA(n_components=3)
    feature_map_rgb = pca.fit_transform(feature_map_flat)

    # Normalize and reshape to image
    feature_map_rgb = feature_map_rgb.reshape(image_height, image_width, 3)
    feature_map_rgb -= feature_map_rgb.min()
    feature_map_rgb /= feature_map_rgb.max()
    left_image = (255 * feature_map_rgb).astype(np.uint8)

    print("left_image size (W x H):", left_image.shape[1], "x", left_image.shape[0])

    Image.fromarray(left_image).save("left_image_output.png")

    
    '''
    vis_base_image = (255 * image_np_hwc).astype(np.uint8)
    
    left_image = #TODO feature_map_chw_proj
    right_image = vis_base_image

    # Add border.
    vis_margin = 8
    dpi = 100
    vis_margin_half = int(0.5 * vis_margin)
    border = 255 * np.ones((left_image.shape[0], vis_margin_half, 3), np.uint8)
    left_image = np.hstack([left_image, border])
    right_image = np.hstack([border, right_image])

    vis_base_util.plot_images(imgs=[left_image, right_image], dpi=dpi)

    tile = vis_base_util.save_plot_to_ndarray()
    
    Image.fromarray(tile).save("crocodebug.png")
    '''

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    main()
