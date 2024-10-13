import os, sys, math
package_dir_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(package_dir_dir)

import glob
import imageio
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
from hamer.models import HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from decord import VideoReader

# from vitpose_model import ViTPoseModel
from DWPose.ControlNet.annotator.dwpose import DWposeDetector

import json
from typing import Dict, Optional


def video2images(video_path):
    out_path=os.path.join(
        os.path.dirname(video_path),
        os.path.splitext(os.path.basename(video_path))[0],
        'images'
        )
    if not os.path.exists(os.path.join(out_path, '0.jpg')):
        os.makedirs(out_path, exist_ok=True)
        reader = imageio.get_reader(video_path)
        for frame_number, im in tqdm(enumerate(reader)):
            out_file = os.path.join(out_path, f'{frame_number}.jpg')
            imageio.imwrite(out_file, im)

def infer_hamer(video_path):
    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    img_folder=os.path.join(os.path.dirname(video_path), base_name, 'images')
    out_folder=os.path.join(os.path.dirname(video_path), base_name, 'hamer')
    os.makedirs(out_folder, exist_ok=True)

    checkpoint=os.path.join(package_dir_dir, 'ckpts/hamer_ckpts/checkpoints/hamer.ckpt')

    # load checkpoints
    model, model_cfg = load_hamer(checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    detectron2_cfg = LazyConfig.load(os.path.join(package_dir_dir, 'hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py'))
    detectron2_cfg.train.init_checkpoint = os.path.join(package_dir_dir, 'ckpts/model_final_f05665.pkl')
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    pose = DWposeDetector()

    image_files = list(Path(img_folder).glob('*.jpg'))
    height,width,_ = cv2.imread(image_files[0]).shape
    for i in tqdm(range(len(image_files))):
        image_name = str(i)+'.jpg'
        image_path = os.path.join(img_folder, image_name)
        out_path = os.path.join(out_folder, image_name)
        if not os.path.exists(out_path):
            img_cv2 = cv2.imread(image_path)
            vitposes_out = pose(img_cv2)
            bboxes = []
            is_right = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes[-42:-21]
                right_hand_keyp = vitposes[-21:]
                # Rejecting not confident detections

                keyp = left_hand_keyp
                if (keyp!=-1).sum()>6: 
                    valid_indices = np.where(keyp != -1)
                    valid_keyp_x = keyp[valid_indices[0], 0]
                    valid_keyp_y = keyp[valid_indices[0], 1]
                    bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                    bboxes.append(bbox)
                    is_right.append(0)
            
                keyp = right_hand_keyp
                if (keyp!=-1).sum()>6: 
                    valid_indices = np.where(keyp != -1)
                    valid_keyp_x = keyp[valid_indices[0], 0]
                    valid_keyp_y = keyp[valid_indices[0], 1]
                    bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) > 0:
                boxes = np.stack(bboxes)
                right = np.stack(is_right)

                # Run reconstruction on all detected hands
                dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

                all_verts = []
                all_cam_t = []
                all_right = []
                
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        out = model(batch)
                    multiplier = (2*batch['right']-1)
                    pred_cam = out['pred_cam']
                    pred_cam[:,1] = multiplier*pred_cam[:,1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    multiplier = (2*batch['right']-1)
                    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                    # Render the result
                    batch_size = batch['img'].shape[0]
                    for n in range(batch_size):
                        # Get filename from path img_path
                        # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                        person_id = int(batch['personid'][n])
                        white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                        input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                        input_patch = input_patch.permute(1,2,0).numpy()

                        regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                batch['img'][n],
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(0, 0, 0),
                                                )
                        hand_mask_img = regression_img
                        hand_img = input_patch

                        # Add all verts and cams to list
                        verts = out['pred_vertices'][n].detach().cpu().numpy()
                        is_right = batch['right'][n].cpu().numpy()
                        verts[:,0] = (2*is_right-1)*verts[:,0]
                        cam_t = pred_cam_t_full[n]
                        all_verts.append(verts)
                        all_cam_t.append(cam_t)
                        all_right.append(is_right)

                # Render front view
                if len(all_verts) > 0:
                    misc_args = dict(
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=scaled_focal_length,
                    )
                    cam_view_diff = renderer.render_rgba_multiple_diff(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                    # Overlay image            
                    cv2.imwrite(out_path, 255*cam_view_diff[:, :, ::-1])
            else:
                black_image = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.imwrite(out_path, black_image)

def infer_hamer_single(image_path):
    LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

    # load checkpoints
    checkpoint=os.path.join(package_dir_dir, 'ckpts/hamer_ckpts/checkpoints/hamer.ckpt')    
    model, model_cfg = load_hamer(checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    detectron2_cfg = LazyConfig.load(os.path.join(package_dir_dir, 'hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py'))
    detectron2_cfg.train.init_checkpoint = os.path.join(package_dir_dir, 'ckpts/model_final_f05665.pkl')
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    pose = DWposeDetector()

    img_cv2 = image_path
    height,width,_ = img_cv2.shape

    vitposes_out = pose(img_cv2)
    bboxes = []
    is_right = []

    # Use hands based on hand keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes[-42:-21]
        right_hand_keyp = vitposes[-21:]
        # Rejecting not confident detections

        keyp = left_hand_keyp
        if (keyp!=-1).sum()>6: 
            valid_indices = np.where(keyp != -1)
            valid_keyp_x = keyp[valid_indices[0], 0]
            valid_keyp_y = keyp[valid_indices[0], 1]
            bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
            bboxes.append(bbox)
            is_right.append(0)
    
        keyp = right_hand_keyp
        if (keyp!=-1).sum()>6: 
            valid_indices = np.where(keyp != -1)
            valid_keyp_x = keyp[valid_indices[0], 0]
            valid_keyp_y = keyp[valid_indices[0], 1]
            bbox = [valid_keyp_x.min(), valid_keyp_y.min(), valid_keyp_x.max(), valid_keyp_y.max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) > 0:
        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(0, 0, 0),
                                        )
                hand_mask_img = regression_img
                hand_img = input_patch

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view_diff = renderer.render_rgba_multiple_diff(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            img_out2 = 255*cam_view_diff[:, :, ::-1]
            img_out2 = img_out2.astype(np.uint8)
            return img_out2
    else:
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        return black_image

def images2video(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    img_folder=os.path.join(os.path.dirname(video_path), base_name, 'images')
    out_folder=os.path.join(os.path.dirname(video_path), base_name, 'hamer')

    image_files = list(Path(img_folder).glob('*.jpg'))
    hamer_files = list(Path(out_folder).glob('*.jpg'))
    mp4_file = os.path.join(os.path.dirname(out_folder), 'hamer.mp4')
    if len(image_files) == len(hamer_files):        
        if not os.path.exists(mp4_file):
            writer = imageio.get_writer(mp4_file, fps=fps)
            image_files = list(Path(img_folder).glob('*.jpg'))
            for i in tqdm(range(len(image_files))):
                im = imageio.imread(os.path.join(out_folder, str(i)+'.jpg'))
                writer.append_data(im)
            writer.close()

video_path='F:/AIGC/000-human/000-smpl/2409-GVHMR/main/docs/example_video/0003.mp4'

# -----------------------------------------------------------------------------------------step1
#video2images(video_path)

# -----------------------------------------------------------------------------------------step2
#infer_hamer(video_path)

# -----------------------------------------------------------------------------------------step3
#images2video(video_path)

print('------------ok')
