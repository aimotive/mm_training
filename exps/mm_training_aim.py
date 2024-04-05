# Copyright (c) Megvii Inc. All rights reserved.
import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import kornia
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data as tdata
import torch.utils.data.distributed
import torchvision
from mmdet3d.core import LiDARInstance3DBoxes
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from scipy.spatial.transform.rotation import Rotation
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

import wandb
from dataset.src.aimotive_dataset import AiMotiveDataset, collate_aim
from exps.conf_aim import *
from models.bev_depth import BEVDepthLiDAR
from utils.eval import MAPCalculator


class BEVDepthLightningModel(LightningModule):

    def __init__(self,
                 data_root=data_root,
                 out_path=out_path,
                 batch_size_per_device=batch_size,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 bda_aug_conf=bda_aug_conf,
                 lidar_conf=lidar_conf,
                 use_cam=use_cam,
                 use_lidar=use_lidar,
                 use_radar=use_radar,
                 look_back=look_back,
                 look_forward=look_forward,
                 fuse_layer_in_channels=fuse_layer_in_channels,
                 use_depth_loss=use_depth_loss,
                 **kwargs):
        super().__init__()
        self.save_dir = Path(out_path) / 'outputs'
        self.use_radar = use_radar
        self.lidar_conf = lidar_conf
        self.use_lidar = use_lidar
        self.use_cam = use_cam
        self.hparams.batch_size = batch_size_per_device
        self.data_root = data_root
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.bda_aug_conf = bda_aug_conf
        self.save_hyperparameters()

        self.model = BEVDepthLiDAR(self.backbone_conf,
                                   self.head_conf,
                                   self.lidar_conf,
                                   is_train_depth=True,
                                   use_lidar=use_lidar,
                                   use_cam=use_cam,
                                   fuse_layer_in_channels=fuse_layer_in_channels)
        self.mode = 'valid'
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2]) + 1
        self.pass_depth_labels = use_depth_loss
        self.look_back = look_back
        self.look_forward = look_forward
        self.map_calculator = MAPCalculator()

        self.eval_split = 'all' if eval_split is None else eval_split

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    @torch.no_grad()
    def augment_images(self, images, depth_images, stage) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, n, c, h, w = images.shape

        if stage != 'train':
            return images, depth_images, np.zeros((b * s * n), dtype=bool)

        images = images.view(b * s * n, c, h, w)
        depth_images = depth_images.permute(0, 3, 1, 2)

        flips = np.random.uniform(size=(b * s * n)) > 0.5

        images = torch.stack(
            [kornia.geometry.transform.hflip(img) if flipped else img for img, flipped in zip(images, flips)]).view(b,
                                                                                                                    s,
                                                                                                                    n,
                                                                                                                    c,
                                                                                                                    h,
                                                                                                                    w)
        depth_images = torch.stack(
            [kornia.geometry.transform.hflip(img) if flipped else img for img, flipped in zip(depth_images, flips)])

        depth_images = depth_images.permute(0, 2, 3, 1).contiguous()

        return images, depth_images, flips

    @torch.no_grad()
    def get_depth_labels(self, images, mats, pointclouds):
        depth_labels = list()

        batch_extrinsics = mats['extrinsics']
        batch_intrinsics = mats['intrin_mats']
        bda_mats = mats['bda_mat']

        for bda_mat, cam_sweep_images, cam_sweep_extrinsics, cam_sweep_intrinsics, pointcloud in zip(bda_mats, images,
                                                                                                     batch_extrinsics,
                                                                                                     batch_intrinsics,
                                                                                                     pointclouds):
            for sweep_images, sweep_extrinsics, sweep_intrinsics in zip(cam_sweep_images, cam_sweep_extrinsics,
                                                                        cam_sweep_intrinsics):
                for extrinsic, intrinsic, image in zip(sweep_extrinsics, sweep_intrinsics, sweep_images):
                    pointcloud_not_augmented = pointcloud.clone()
                    pointcloud_not_augmented[:, :3] = pointcloud_not_augmented[:, :3] @ torch.linalg.inv(
                        bda_mat[:3, :3]).T

                    depth_image = self.get_depth_image(extrinsic, image, intrinsic, pointcloud_not_augmented)
                    depth_labels.append(depth_image.unsqueeze(0))

        depth_labels = torch.stack(depth_labels).unsqueeze(1)
        B, S, C, _, H, W = images.shape
        depth_labels = depth_labels.reshape(B, S * C, H, W)
        depth_labels_downsampled = self.get_downsampled_gt_depth(depth_labels)
        return depth_labels_downsampled

    @torch.no_grad()
    def get_depth_image(self, extrinsic, image, intrinsic, pointcloud):
        points = torch.ones_like(pointcloud[:, 0:4])
        points[:, 0:3] = pointcloud[:, 0:3]
        points = points.T
        points = extrinsic @ points
        depths = points[2, :]
        points_projected = intrinsic @ points
        points_projected = points_projected / points_projected[2:3, :]
        mask = torch.ones_like(depths, dtype=torch.bool)
        mask = torch.logical_and(mask, depths > 1.0)
        mask = torch.logical_and(mask, points_projected[0, :] > 1)
        mask = torch.logical_and(mask, points_projected[0, :] < image.shape[2] - 1)
        mask = torch.logical_and(mask, points_projected[1, :] > 1)
        mask = torch.logical_and(mask, points_projected[1, :] < image.shape[1] - 1)
        points_projected = points_projected[:, mask]
        depths = depths[mask]
        points_projected = points_projected.to(torch.long)
        depth_map = torch.zeros_like(image)[0]
        depth_map[points_projected[1], points_projected[0]] = depths

        return depth_map

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = depth_labels.view(-1, self.depth_channels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        # gt_depths_tmp = gt_depths.mean(-1).mean(-1)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        # Is taking the minimum point valid?
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels)
        return gt_depths.float()
        # return gt_depths.float()[..., 1:]

    def pred_dict_to_aimotive_dict(self, pred_dict):
        out_dict = dict()
        out_dict['CapturedObjects'] = list()

        pred = pred_dict['pred_boxes']
        scores = pred_dict['pred_scores']
        classes = [self.class_names[c] for c in pred_dict['pred_labels']]

        for (obj_pred, obj_score, obj_class) in zip(pred, scores, classes):
            obj = dict()
            obj['BoundingBox3D Origin X'] = float(obj_pred[0])
            obj['BoundingBox3D Origin Y'] = float(obj_pred[1])
            obj['BoundingBox3D Origin Z'] = float(obj_pred[2]) + (float(obj_pred[5]) / 2)
            obj['BoundingBox3D Extent X'] = float(obj_pred[3])
            obj['BoundingBox3D Extent Y'] = float(obj_pred[4])
            obj['BoundingBox3D Extent Z'] = float(obj_pred[5])

            quat = Rotation.from_euler('z', [obj_pred[6]], degrees=False).as_quat()
            obj['BoundingBox3D Relative Velocity X'] = float(obj_pred[7])
            obj['BoundingBox3D Relative Velocity Y'] = float(obj_pred[8])
            obj['BoundingBox3D Relative Velocity Z'] = 0

            obj['BoundingBox3D Orientation Quat X'] = float(quat[0][0])
            obj['BoundingBox3D Orientation Quat Y'] = float(quat[0][1])
            obj['BoundingBox3D Orientation Quat Z'] = float(quat[0][2])
            obj['BoundingBox3D Orientation Quat W'] = float(quat[0][3])

            obj['ObjectType'] = obj_class
            obj['Score'] = float(obj_score)

            out_dict['CapturedObjects'].append(obj)

        return out_dict

    def training_step(self, batch, batch_idx):
        (sweep_imgs, mats, img_metas, pointclouds, gt_boxes, gt_labels) = batch

        if self.use_cam:
            depth_labels = self.get_depth_labels(sweep_imgs, mats, pointclouds)
            sweep_imgs = self.normalize_images(sweep_imgs)
            sweep_imgs, depth_labels, flipped = self.augment_images(sweep_imgs, depth_labels, 'train')
            mats['flipped'] = flipped
            input_depth = depth_labels.permute(0, 3, 1, 2) if self.pass_depth_labels else None
            if len(depth_labels.shape) == 5:
                # only key-frame will calculate depth loss
                depth_labels = depth_labels[:, 0, ...]
        else:
            input_depth = None
            depth_labels = None

        preds, depth_preds, lidar_bev_ret, cam_bev_ret = self.model((sweep_imgs, pointclouds), mats, input_depth)

        if batch_idx % 200 == 0 and isinstance(self.logger, WandbLogger):
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                results = self.model.module.get_bboxes(preds, img_metas)
            else:
                results = self.model.get_bboxes(preds, img_metas)

        detection_loss, targets = self.get_targets_and_detection_loss(gt_boxes, gt_labels, preds)

        depth_loss = 0.0
        if self.use_cam:
            depth_loss = self.get_depth_loss(depth_labels, depth_preds)

        if batch_idx % 200 == 0 and isinstance(self.logger, WandbLogger):
            self.log_pointcloud(pointclouds[0], gt_boxes[0], gt_labels[0], results[0])
            self.log_images(depth_labels, depth_preds, preds, targets)

        self.log('train_detection_loss', detection_loss, prog_bar=True, sync_dist=True)
        self.log('train_depth_loss', depth_loss, prog_bar=True, sync_dist=True)
        self.log('train_loss', detection_loss + depth_loss, prog_bar=True, sync_dist=True)
        return {'loss': detection_loss + depth_loss, 'detection_loss': detection_loss, 'depth_loss': depth_loss}

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, img_metas, pointclouds, gt_boxes, gt_labels) = batch

        if self.use_cam:
            depth_labels = self.get_depth_labels(sweep_imgs, mats, pointclouds)
            sweep_imgs = self.normalize_images(sweep_imgs)
            sweep_imgs, depth_labels, flipped = self.augment_images(sweep_imgs, depth_labels, 'val')
            mats['flipped'] = flipped
            input_depth = depth_labels.permute(0, 3, 1, 2) if self.pass_depth_labels else None
            if len(depth_labels.shape) == 5:
                # only key-frame will calculate depth loss
                depth_labels = depth_labels[:, 0, ...]
        else:
            input_depth = None
            depth_labels = None

        
        preds, depth_preds, lidar_bev_ret, cam_bev_ret = self.model((sweep_imgs, pointclouds), mats, input_depth)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)

        result_list = [[r[0].tensor.cpu().detach().numpy(), r[1].cpu().detach().numpy(), r[2].cpu().detach().numpy()] for r in results]
        gt_boxes_cpu  = list(map(lambda x: x.cpu().detach().numpy(), gt_boxes))
        gt_labels_cpu = list(map(lambda x: x.cpu().detach().numpy(), gt_labels))

        target_list = [[gt_box, gt_label] for gt_box, gt_label in zip(gt_boxes_cpu, gt_labels_cpu)]

        self.map_calculator.update(target_list, result_list, [img_meta['path'] for img_meta in img_metas])
        if prefix == 'test':
            result_dict = self.result_list_to_dict(result_list)

            out = list(map(self.pred_dict_to_aimotive_dict, result_dict))
            self.save_results(out, img_metas)

        detection_loss, targets = self.get_targets_and_detection_loss(gt_boxes, gt_labels, preds)

        if batch_idx % 200 == 0 and isinstance(self.logger, WandbLogger):
            self.log_pointcloud(pointclouds[0], gt_boxes[0], gt_labels[0], results[0])
            self.log_images(depth_labels, depth_preds, preds, targets)

        depth_loss = 0
        if self.use_cam:
            depth_loss = self.get_depth_loss(depth_labels, depth_preds)

        self.log(f'{prefix}_detection_loss', detection_loss, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_depth_loss', depth_loss, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}_loss', detection_loss + depth_loss, prog_bar=True, sync_dist=True)

        return {'loss': detection_loss + depth_loss, 'detection_loss': detection_loss, 'depth_loss': depth_loss}

    def predict_step(self, batch, batch_idx):
        (sweep_imgs, mats, img_metas, pointclouds, gt_boxes, gt_labels) = batch

        if self.use_cam:
            depth_labels = self.get_depth_labels(sweep_imgs, mats, pointclouds)
            sweep_imgs = self.normalize_images(sweep_imgs)
            sweep_imgs, depth_labels, flipped = self.augment_images(sweep_imgs, depth_labels, 'val')
            mats['flipped'] = flipped
            input_depth = depth_labels.permute(0, 3, 1, 2) if self.pass_depth_labels else None
        else:
            input_depth = None

        preds, depth_preds, lidar_bev_ret, cam_bev_ret = self.model((sweep_imgs, pointclouds), mats, input_depth)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)

        result_list = [[r[0].tensor.cpu().detach().numpy(), r[1].cpu().detach().numpy(), r[2].cpu().detach().numpy()]
                       for r in results]

        result_dict = self.result_list_to_dict(result_list)

        out = list(map(self.pred_dict_to_aimotive_dict, result_dict))
        self.save_results(out, img_metas)

    def result_list_to_dict(self, result_list):
        new_result_list = list()
        for result_per_sample in result_list:
            result_per_sample_dict = {'pred_boxes': result_per_sample[0], 'pred_scores': result_per_sample[1],
                                      'pred_labels': result_per_sample[2]}
            new_result_list.append(result_per_sample_dict)
        return new_result_list

    def log_pointcloud(self, pc, gt_boxes, gt_labels, results):
        def create_box_list_from_output(corners, labels, color):
            box_list = list()
            for box, label in zip(corners, labels):
                box_dict = {"corners": box.tolist(),
                            "label": CLASSES[int(label)],
                            "color": color}
                box_list.append(box_dict)
            return box_list

        gt_boxes[:, 2] -= (gt_boxes[:, 5] / 2.)
        gt_boxes_corners = LiDARInstance3DBoxes(gt_boxes[:, :7]).corners.cpu().detach().numpy()
        pred_boxes_corners = results[0].corners.cpu().detach().numpy()
        pred_labels = results[2]
        gt_boxes_list = create_box_list_from_output(gt_boxes_corners, gt_labels, [255, 255, 255])
        pred_boxes_list = create_box_list_from_output(pred_boxes_corners, pred_labels, [255, 0, 0])

        try:
            point_scene = wandb.Object3D({
                "type": "lidar/beta",
                "points": pc.cpu().detach().numpy(),
                "boxes": np.array(gt_boxes_list + pred_boxes_list)
            })
            wandb.log({"point_scene": point_scene})
        except wandb.errors.Error:
            print("Couldnt log point cloud")


    def inv_sigmoid(self, x):
        return -torch.log((1 / (x + 1e-8)) - 1)

    def on_validation_epoch_end(self) -> None:
        self.eval_end('val')

    def on_test_epoch_end(self) -> None:
        self.eval_end('test')

    def eval_end(self, prefix: str):
        eval_output_dict = self.map_calculator.compute_bev(iou_thr=0.3, x_range=point_cloud_range[3])

        for key in eval_output_dict:
            self.log(f'{prefix}_{key}', eval_output_dict[key], prog_bar=True, sync_dist=True)

        self.map_calculator.reset()

    @torch.no_grad()
    def test_time_augment(self, x, mats_dict, lidar_oracle=None, timestamps=None):
        preds = None
        img_og, lidar_og = x
        for x_flip in [1, -1]:
            for y_flip in [1, -1]:
                img = img_og.clone()
                lidar = [l.clone() for l in lidar_og]
                flip_mat = torch.FloatTensor([[x_flip * 1, 0, 0, 0],
                                              [0, y_flip * 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]]).to(device=img.device)

                lidar_new = list()
                for l in lidar:
                    l[:, :3] = l[:, :3] @ flip_mat[:3, :3].T
                    lidar_new.append(l)
                lidar = lidar_new

                flip_mat = flip_mat.view(1, 4, 4).repeat(len(lidar), 1, 1)

                if preds is None:
                    preds, depth_preds, lidar_bev_ret, cam_bev_ret = self.model((img, lidar), mats_dict, lidar_oracle)

                    for pred in preds:
                        pred[0]['heatmap'] = torch.sigmoid(pred[0]['heatmap'])
                        pred[0]['dim'] = torch.exp(pred[0]['dim'])

                else:
                    preds_current, depth_preds, lidar_bev_ret, cam_bev_ret = self.model((img, lidar), mats_dict,
                                                                                        lidar_oracle)

                    flip_mat_inv = torch.linalg.inv(flip_mat)
                    for pred_class, pred_class_current in zip(preds, preds_current):

                        # for preds_sweep, preds_sweep_current in zip(pred_class[0], pred_class_current[0]):
                        for key in pred_class[0]:
                            out = self.model.bev_augment_image(pred_class_current[0][key], flip_mat_inv)
                            if key == 'heatmap':
                                out = torch.sigmoid(out)
                            if key == 'dim':
                                out = torch.exp(out)
                            if key == 'vel':
                                out = (flip_mat_inv[0, :2, :2] @ out.permute(1, 2, 3, 0).contiguous().view(2, 256*256 * 4)).view(2, 256, 256, 4).permute(3, 0, 1, 2)
                            if key == 'rot':
                                rot_sine = out[:, 0, :, :]
                                rot_cosine = out[:, 1, :, :]
                                rot = torch.atan2(rot_sine, rot_cosine)
                                if x_flip == -1:
                                    rot = rot - 2 * torch.asin(torch.tensor(1.0))
                                if y_flip == -1:
                                    rot = -rot

                                out[..., 0, :, :] = rot.sin()
                                out[..., 1, :, :] = rot.cos()
                            pred_class[0][key] += out

        for pred_class in preds:
            for key in pred_class[0]:
                pred_class[0][key] /= 4
                if key == 'heatmap':
                    pred_class[0][key] = self.inv_sigmoid(pred_class[0][key])
                if key == 'dim':
                    pred_class[0][key] = torch.log(pred_class[0][key])

        return preds, depth_preds, lidar_bev_ret, cam_bev_ret

    def log_images(self, depth_labels, depth_preds, preds, targets):
        self.logger.log_image(key="Predictions", images=list(preds[0][0]['heatmap']))
        self.logger.log_image(key="Targets", images=list(targets[0][0]))
        if self.use_cam:
            self.logger.log_image(key="Depth predictions", images=list(depth_preds[:4].argmax(1) / 100),
                                  caption=['front', 'back'])
            self.logger.log_image(key="Depth target",
                                  images=list(depth_labels.permute(0, 3, 1, 2)[:4].argmax(1) / 100),
                                  caption=['front', 'back'])

    def get_targets_and_detection_loss(self, gt_boxes, gt_labels, preds):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        return detection_loss, targets

    def normalize_images(self, sweep_imgs):
        sweep_imgs = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(sweep_imgs[:, :, :, :3, ...] / 255.)
        return sweep_imgs

    def save_results(self, results, img_metas):
        for img_meta, result in zip(img_metas, results):
            save_path = img_meta['path'].replace(self.data_root, str(self.save_dir))
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(result, f)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        lr = learning_rate # 3e-4

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = AiMotiveDataset(self.data_root, point_cloud_range, split='train', bda_aug_conf=bda_aug_conf,
                                        use_cam=self.use_cam, use_lidar=self.use_lidar,
                                        use_radar=self.use_radar, look_back=self.look_back,
                                        look_forward=self.look_forward)
        train_loader = tdata.DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_aim
        )
        return train_loader

    def predict_dataloader(self):
        return self.val_dataloader()

    def val_dataloader(self):
        val_dataset = AiMotiveDataset(self.data_root, point_cloud_range, split='val', bda_aug_conf=bda_aug_conf,
                                      use_cam=self.use_cam, use_lidar=self.use_lidar,
                                      use_radar=self.use_radar, look_back=self.look_back,
                                      look_forward=self.look_forward, eval_odd=self.eval_split)
        val_loader = tdata.DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_aim
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser


def create_trainer(args: Namespace) -> pl.Trainer:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    output_path = Path(out_path)

    output_path.mkdir(exist_ok=True, parents=True)
    current_path = os.path.dirname(os.path.abspath(__file__))
    shutil.copyfile(os.path.join(current_path, 'conf_aim.py'), str(output_path / 'config.py'))

    model_save_path = output_path / 'saved_models'

    logger = WandbLogger(project="mm_training_exp", name=experiment_name, entity='aimotive') if log_wandb else TensorBoardLogger(output_path, name=experiment_name)


    save_best_callback = ModelCheckpoint(dirpath=str(model_save_path),
                                         filename='{epoch}-{step}-{val_detection_loss:.2f}',
                                         monitor="val_detection_loss",
                                         save_top_k=10)

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[
                                                EarlyStopping(monitor="val_detection_loss", mode="min", patience=8),
                                                save_best_callback,
                                                ModelCheckpoint(dirpath=str(model_save_path),
                                                                filename='latest-{epoch}-{step}',
                                                                monitor="step",
                                                                every_n_train_steps=500,
                                                                save_top_k=1),
                                            ],
                                            default_root_dir=output_path,
                                            logger=logger,
                                            )
    return trainer

def parse_arguments():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)

    parser.set_defaults(
        deterministic=False,
        max_epochs=999,
        log_every_n_steps=50,
        accelerator='gpu',
        num_sanity_val_steps=2,
        benchmark=True,
        gradient_clip_val=2,
        **trainer_params
    )

    args = parser.parse_args()
    return args

def main_train():
    args = parse_arguments()
    model = BEVDepthLightningModel(**vars(args))
    trainer = create_trainer(args)

    trainer.fit(model, ckpt_path=ckpt_path) if ckpt_path is not None else trainer.fit(model)
    trainer.test(model, ckpt_path=trainer.callbacks[4].best_model_path)


if __name__ == '__main__':
    main_train()
