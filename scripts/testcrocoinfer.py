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


    # ---------------------------------- #
    # ----- image and vis settings ----- #
    # ---------------------------------- #
    # 0392, 0434
    debug_strategy2 = True
    query_template_id = 392
    object_id = 1
    dataset_type = "lmo"
    vis_for_paper = [False, True]      # False for detailed debug images, True for having the tiled images with feature maps
    bg_noise = False
    bg_realimage = True
    saveplots = False
    #resize_value = 224

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savefldr = os.path.join("debug", f"run_{timestamp}")
    if saveplots:
        os.makedirs(savefldr, exist_ok=True)
    # ---------------------------------- #
    # ----- image and vis settings ----- #
    # ---------------------------------- #


    query_image = np.array(Image.open(f"bop_datasets/templates/v1/{dataset_type}/{object_id}/rgb/template_0{query_template_id}.png"))
    original_height, original_width = query_image.shape[:2]  # shape = (H, W, C)
    print(original_height, original_width)

    if bg_noise:
        # Create random noise background
        random_background = np.random.randint(0, 256, size=query_image.shape, dtype=np.uint8)

        # Mask out the black parts
        mask = np.all(query_image == [0, 0, 0], axis=-1)

        # Replace black pixels with the random background
        query_image_with_bg = query_image.copy()
        query_image_with_bg[mask] = random_background[mask]
        query_image = query_image_with_bg

    elif bg_realimage:
        # Load a background image and resize to match your query image
        bg_img = np.array(Image.open("debug/bg.png").resize((original_width, original_height)))

        # Mask for black background in query image
        mask = np.all(query_image == [0, 0, 0], axis=-1)

        # Replace black background with pixels from background image
        query_image_with_bg = query_image.copy()
        query_image_with_bg[mask] = bg_img[mask]
        query_image = query_image_with_bg


    # Resize image because we want it to be 224x224
    #query_image = cv2.resize(query_image, (resize_value, resize_value), interpolation=cv2.INTER_LINEAR)
    #resize_scale_x = resize_value / original_width   # for later focal length adjustment
    #resize_scale_y = resize_value / original_height  # for later focal length adjustment

    image_np_hwc = query_image.astype(np.float32)/255.0

    # Extract feature map.
    image_tensor_chw = array_to_tensor(image_np_hwc).to(torch.float32).permute(2,0,1).to(device)
    image_tensor_bchw = image_tensor_chw.unsqueeze(0)
    extractor_output = extractor(image_tensor_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]

    # Load object representation
    repre = repre_util.load_object_repre(
    repre_dir=f"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/object_repre/{dataset_type}/crocov2/{object_id}",
    tensor_device=device,
    )

    # ------ start debug strategy 2: remove query template from pool (and its registered features), and so on..
    if debug_strategy2:
        # store camera before filtering (used later)
        query_template_camera_c2w = repre.template_cameras_cam_from_model[query_template_id]
        # remove features belonging to the query template
        
        if bg_noise:
            excluded_template_ids = [715, 144, 504]
            logger.info(f"Debug method: Strategy 2 + Noise Background. Removing: {excluded_template_ids}")
        elif bg_realimage:
            excluded_template_ids = [query_template_id, 574, 642]
            logger.info(f"Debug method: Strategy 2 + RealImage Background. Removing: {excluded_template_ids}")
        else:
            excluded_template_ids = [query_template_id, 280, 403, 402, 401]  # query_template_id, 280, 403, 402, 401
            logger.info(f"Debug method: Strategy 2. Removing: {excluded_template_ids}")
        
        excluded_template_ids_tensor = torch.tensor(excluded_template_ids, device=repre.feat_to_template_ids.device)
        valid_feat_mask = ~torch.isin(repre.feat_to_template_ids, excluded_template_ids_tensor)
        valid_feat_mask = valid_feat_mask.to(repre.feat_to_vertex_ids.device)
        repre.feat_vectors = repre.feat_vectors[valid_feat_mask]
        repre.feat_to_template_ids = repre.feat_to_template_ids[valid_feat_mask]
        repre.feat_to_vertex_ids = repre.feat_to_vertex_ids[valid_feat_mask]
        repre.feat_to_cluster_ids = repre.feat_to_cluster_ids[valid_feat_mask]
        # Remove template_cameras_cam_from_model (list) — pop in reverse order to avoid index shifting
        for tid in sorted(excluded_template_ids, reverse=True):
            repre.template_cameras_cam_from_model.pop(tid)
        # Remove from template_descs (tensor) — use masking to keep only valid indices
        keep_indices = [i for i in range(repre.template_descs.shape[0]) if i not in excluded_template_ids]
        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long, device=repre.template_descs.device)
        repre.template_descs = repre.template_descs[keep_indices_tensor]
    # ------ end debug strategy 2:

    # Build a kNN index from object feature vectors.
    visual_words_knn_index = None
    if opts.match_template_type == "tfidf":
        visual_words_knn_index = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric
        )
        visual_words_knn_index.fit(repre.feat_cluster_centroids)

        # Build per-template KNN index with features from that template.
        template_knn_indices = []
        if opts.match_feat_matching_type == "cyclic_buddies":
            logger.info("Building per-template KNN indices...")
            for template_id in range(len(repre.template_cameras_cam_from_model)):
                logger.info(f"Building KNN index for template {template_id}...")
                tpl_feat_mask = repre.feat_to_template_ids == template_id
                tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

                template_feats = repre.feat_vectors[tpl_feat_ids]

                # Build knn index for object features.
                template_knn_index = knn_util.KNN(k=1, metric="l2")
                template_knn_index.fit(template_feats.cpu())
                template_knn_indices.append(template_knn_index)
            logger.info("Per-template KNN indices built.")

        # Keep only points inside the object mask.
        mask_path = f"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/templates/v1/{dataset_type}/{object_id}/mask/template_0{query_template_id}.png"
        mask_image_arr = inout.load_im(mask_path)
        mask_modal_tensor = array_to_tensor(mask_image_arr).to(device)

        query_points = feature_util.filter_points_by_mask(
            grid_points, mask_modal_tensor
        )

        # Extract features at the selected points, of shape (num_points, feat_dims).
        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
        ).contiguous()

        # Potentially project features to a PCA space.
        if (
            query_features.shape[1] != repre.feat_vectors.shape[1]
            and len(repre.feat_raw_projectors) != 0
        ):
            query_features_proj = projector_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()

            _c, _h, _w = feature_map_chw.shape
            feature_map_chw_proj = (
                projector_util.project_features(
                    feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                    projectors=repre.feat_raw_projectors,
                )
                .view(_h, _w, -1)
                .permute(2, 0, 1)
            )
        else:
            query_features_proj = query_features
            feature_map_chw_proj = feature_map_chw

        '''
        feature_map_chw_proj_vis = vis_util.vis_pca_feature_map(
            feature_map_chw=feature_map_chw,
            image_height=resize_value,
            image_width=resize_value,
            pca_projector=repre.feat_vis_projectors[0],
        )
        Image.fromarray(feature_map_chw_proj_vis).save(f"query_feat_vis_{resize_value}.png")
        '''
            
        # Establish 2D-3D correspondences.
        corresp = []
        if len(query_points) != 0:
            corresp = corresp_util.establish_correspondences(
                query_points=query_points,
                query_features=query_features_proj,
                object_repre=repre,
                template_matching_type=opts.match_template_type,
                template_knn_indices=template_knn_indices,
                feat_matching_type=opts.match_feat_matching_type,
                top_n_templates=opts.match_top_n_templates,
                top_k_buddies=opts.match_top_k_buddies,
                visual_words_knn_index=visual_words_knn_index,
                debug=opts.debug,
            )

            best_template = max(corresp, key=lambda x: x["template_score"])
            best_template_id = best_template["template_id"]
            best_template_score = best_template["template_score"]
            logger.info(f"Best template: {best_template_id}, with score: {best_template_score}")

            # Estimate coarse poses from corespondences.
            coarse_poses = []
            for corresp_id, corresp_curr in enumerate(corresp):

                # We need at least 3 correspondences for P3P.
                num_corresp = len(corresp_curr["coord_2d"])
                if num_corresp < 6:
                    logger.info(f"Only {num_corresp} correspondences, skipping.")
                    continue
                
                # get the camera model for the query template of interest
                if debug_strategy2:
                    camera_c2w = query_template_camera_c2w
                else:
                    camera_c2w = repre.template_cameras_cam_from_model[query_template_id]

                # Now we adjust the focal length due to the image resize into 224x224
                # Create updated camera intrinsics due to resize
                '''
                fx, fy = camera_c2w.f
                cx, cy = camera_c2w.c
                new_fx = fx * resize_scale_x
                new_fy = fy * resize_scale_y
                new_cx = cx * resize_scale_x
                new_cy = cy * resize_scale_y
                # Rebuild the camera model with new intrinsics
                camera_c2w = PinholePlaneCameraModel(
                    f=(new_fx, new_fy),
                    c=(new_cx, new_cy),
                    width=resize_value,
                    height=resize_value,
                    T_world_from_eye=camera_c2w.T_world_from_eye  # reuse original pose
                )
                # end of adjustment
                '''

                (
                    coarse_pose_success,
                    R_m2c_coarse,
                    t_m2c_coarse,
                    inliers_coarse,
                    quality_coarse,
                ) = pnp_util.estimate_pose(
                    corresp=corresp_curr,
                    camera_c2w=camera_c2w,
                    pnp_type=opts.pnp_type,
                    pnp_ransac_iter=opts.pnp_ransac_iter,
                    pnp_inlier_thresh=opts.pnp_inlier_thresh,
                    pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
                    pnp_refine_lm=opts.pnp_refine_lm,
                )

                logger.info(
                    f"Quality of coarse pose {corresp_id}: {quality_coarse}"
                )

                if coarse_pose_success:
                    coarse_poses.append(
                        {
                            "type": "coarse",
                            "R_m2c": R_m2c_coarse,
                            "t_m2c": t_m2c_coarse,
                            "corresp_id": corresp_id,
                            "quality": quality_coarse,
                            "inliers": inliers_coarse,
                        }
                    )

            # Find the best coarse pose.
            best_coarse_quality = None
            best_coarse_pose_id = 0
            for coarse_pose_id, pose in enumerate(coarse_poses):
                if (
                    best_coarse_quality is None
                    or pose["quality"] > best_coarse_quality
                ):
                    best_coarse_pose_id = coarse_pose_id
                    best_coarse_quality = pose["quality"]

            # Select the final pose estimate.
            final_poses = []
            
            if opts.final_pose_type in [
                "best_coarse",
            ]:

                # If no successful coarse pose raise error.
                assert len(coarse_poses) != 0

                # Select the refined pose corresponding to the best coarse pose as the final pose.
                final_pose = None

                if opts.final_pose_type in [
                    "best_coarse",
                ]:
                    final_pose = coarse_poses[best_coarse_pose_id]

                if final_pose is not None:
                    final_poses.append(final_pose)

            else:
                raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

            # Iterate over the final poses to collect visuals.
            for hypothesis_id, final_pose in enumerate(final_poses):

                # Visualizations and saving of results.
                vis_tiles = []

                # Increment hypothesis id by one for each found pose hypothesis.
                pose_m2w = None
                pose_m2w_coarse = None

                # Express the estimated pose as an m2w transformation.
                pose_est_m2c = structs.ObjectPose(
                    R=final_pose["R_m2c"], t=final_pose["t_m2c"]
                )
                trans_c2w = camera_c2w.T_world_from_eye

                trans_m2w = trans_c2w.dot(misc.get_rigid_matrix(pose_est_m2c))
                pose_m2w = structs.ObjectPose(
                    R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:]
                )

                # Get image for visualization.
                vis_base_image = (255 * image_np_hwc).astype(np.uint8)

                # Convert correspondences from tensors to numpy arrays.
                best_corresp_np = tensors_to_arrays(
                    corresp[final_pose["corresp_id"]]
                )

                # IDs and scores of the matched templates.
                matched_template_ids = [c["template_id"] for c in corresp]
                matched_template_scores = [c["template_score"] for c in corresp]

                # Skip evaluation if there is no ground truth available, and only keep
                # the estimated poses.
                pose_eval_dict = None
                pose_eval_dict_coarse = None

                # -- debug: spoof GT annotation of the template of interest:
                from utils.structs import ObjectAnnotation
                from utils.structs import ObjectPose
                import json
                
                with open(f"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/templates/v1/{dataset_type}/{object_id}/metadata.json", "r") as f:
                    metadata = json.load(f)

                try:
                    entry = next(item for item in metadata if item["template_id"] == query_template_id)
                except StopIteration:
                    raise ValueError(f"Template ID {template_id} not found in metadata.")
                

                # Extract pose (R and t)
                R = np.array(entry["pose"]["R"], dtype=np.float32)
                t = np.array(entry["pose"]["t"], dtype=np.float32).reshape(3, 1)
                
                pose_gt = ObjectPose(
                    R=R,
                    t=t
                )

                mask_modal = mask_image_arr

                instance = {
                    "gt_anno": ObjectAnnotation(
                        dataset=dataset_type,
                        lid=object_id,
                        pose=pose_gt,
                        masks_modal=mask_modal
                    ),
                    "input_mask_modal": mask_modal,
                }

                # Get the object mesh and meta information.
                model_path = f"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/{dataset_type}/models/obj_00000{object_id}.ply"
                object_mesh = inout.load_ply(model_path)
                models_info = inout.load_json(f"/home/tatiana/chris-sem-prj/ETH-Semester-Project/bop_datasets/{dataset_type}/models/models_info.json", keys_to_int=True)
                object_lid = object_id
                object_syms = bop_misc.get_symmetry_transformations(
                    models_info[object_lid], max_sym_disc_step= 0.01
                )
                object_diameter = models_info[object_lid]["diameter"]

                max_vertices = 1000
                subsampled_vertices = np.random.permutation(object_mesh["pts"])[:max_vertices]

                bop_chunk_id = 2
                bop_im_id = 8
                inst_j = 0
                orig_camera_c2w = camera_c2w
                times = {'prep': 0.18827414512634277, 
                         'feat_extract': 0.4612290859222412, 
                         'grid_sample': 0.008441686630249023, 
                         'proj': 0.0050792694091796875, 
                         'corresp': 0.09356403350830078, 
                         'pose_coarse': 0.09729123115539551, 
                         'final_select': 8.153915405273438e-05}
                
                pose_evaluator = eval_util.EvaluatorPose([object_lid])

                repre_np = repre_util.convert_object_repre_to_numpy(repre)

                renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
                renderer = renderer_builder.build(renderer_type=renderer_type, model_path=model_path)

                from utils.structs import AlignedBox2f
                def get_box_from_mask(mask: np.ndarray) -> List[int]:
                    """Computes [left, top, right, bottom] from binary mask."""
                    y_coords, x_coords = np.where(mask > 0)
                    if len(x_coords) == 0 or len(y_coords) == 0:
                        return [0, 0, 0, 0]  # Handle empty masks safely
                    left = np.min(x_coords)
                    right = np.max(x_coords)
                    top = np.min(y_coords)
                    bottom = np.max(y_coords)
                    return [left, top, right, bottom]

                left, top, right, bottom = get_box_from_mask(mask_modal)
                box_amodal = AlignedBox2f(left, top, right, bottom)
                # -- end debug

                if instance["gt_anno"] is not None:

                    retrieved_templates_camera_m2c = [
                        repre.template_cameras_cam_from_model[tpl_id]
                        for tpl_id in matched_template_ids
                    ]

                    pose_eval_dict = pose_evaluator.update(
                        scene_id=bop_chunk_id,
                        im_id=bop_im_id,
                        inst_id=inst_j,
                        hypothesis_id=hypothesis_id,
                        base_image=vis_base_image,
                        object_repre_vertices=tensor_to_array(repre.vertices),
                        obj_lid=object_lid,
                        object_pose_m2w=pose_m2w,
                        object_pose_m2w_gt=instance["gt_anno"].pose,
                        orig_camera_c2w=orig_camera_c2w,
                        camera_c2w=camera_c2w,
                        pred_mask=instance["input_mask_modal"],
                        gt_mask=instance["gt_anno"].masks_modal,
                        corresp=best_corresp_np,
                        retrieved_templates_camera_m2c=retrieved_templates_camera_m2c,
                        time_per_inst=times,
                        inlier_radius=opts.pnp_inlier_thresh,
                        object_mesh_vertices=object_mesh["pts"],
                        object_syms=object_syms,
                        object_diameter=object_diameter,
                    )


                else:
                    pose_eval_dict = pose_evaluator.update_without_anno(
                        scene_id=bop_chunk_id,
                        im_id=bop_im_id,
                        inst_id=inst_j,
                        hypothesis_id=hypothesis_id,
                        object_repre_vertices=tensor_to_array(repre.vertices),
                        obj_lid=object_lid,
                        object_pose_m2w=pose_m2w,
                        orig_camera_c2w=orig_camera_c2w,
                        camera_c2w=orig_camera_c2w,
                        time_per_inst=times,
                        corresp=best_corresp_np,
                        inlier_radius=(opts.pnp_inlier_thresh),
                    )
                    

                object_pose_m2w_gt = None
                if "gt_anno" in instance and instance["gt_anno"] is not None:
                    object_pose_m2w_gt = instance["gt_anno"].pose
                

                # Optionally visualize the results.
                if opts.vis_results:

                    # IDs and scores of the matched templates.
                    matched_template_ids = [c["template_id"] for c in corresp]
                    matched_template_scores = [c["template_score"] for c in corresp]

                    if saveplots:
                        for i in range(2):
                            vis_tiles += vis_util.vis_inference_results(
                                base_image=vis_base_image,
                                object_repre=repre_np,
                                object_lid=object_lid,
                                object_pose_m2w=pose_m2w, # pose_m2w,
                                object_pose_m2w_gt=object_pose_m2w_gt,
                                feature_map_chw=feature_map_chw,
                                feature_map_chw_proj=feature_map_chw_proj,
                                vis_feat_map=opts.vis_feat_map,
                                object_box=box_amodal.array_ltrb(),
                                object_mask=mask_modal,
                                camera_c2w=camera_c2w,
                                corresp=best_corresp_np,
                                matched_template_ids=matched_template_ids,
                                matched_template_scores=matched_template_scores,
                                best_template_ind=final_pose["corresp_id"],
                                renderer=renderer,
                                pose_eval_dict=pose_eval_dict,
                                corresp_top_n=opts.vis_corresp_top_n,
                                inlier_thresh=(opts.pnp_inlier_thresh),
                                object_pose_m2w_coarse=pose_m2w_coarse,
                                pose_eval_dict_coarse=pose_eval_dict_coarse,
                                # For paper visualizations:
                                vis_for_paper=vis_for_paper[i],
                                extractor=extractor,
                                debug_croco=True
                            )

                            # Assemble visualization tiles to a grid and save it.
                            if len(vis_tiles):
                                if repre.feat_vis_projectors[0].pca.n_components == 12:
                                    pca_tiles = np.vstack(vis_tiles[1:5])
                                    vis_tiles = np.vstack([vis_tiles[0]] + vis_tiles[5:])
                                    vis_grid = np.hstack([vis_tiles, pca_tiles])
                                else:
                                    vis_grid = np.vstack(vis_tiles)
                                text = {0: "_detailed", 1: ""}
                                vis_path = os.path.join(savefldr, f"result_vis{text[i]}.png")
                                inout.save_im(vis_path, vis_grid)
                                logger.info(f"Visualization saved to {vis_path}")

                                vis_tiles = []

    # Empty unused GPU cache variables.
    if device == "cuda":
        time_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        time_end = time.time()
        logger.info(f"Garbage collection took {time_end - time_start} seconds.")

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(
        InferOpts
    )[0]
    infer(opts)


if __name__ == "__main__":
    main()
