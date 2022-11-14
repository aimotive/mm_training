import os
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
import torch.utils.data as tdata
from mmdet3d.core import LiDARInstance3DBoxes

from dataset.src.data_loader import DataLoader, DataItem
from dataset.src.sequence import Sequence

CATEGORY_MAPPING = {'CAR': 0, 'Size_vehicle_m': 0,
                    'TRUCK': 1, 'BUS': 1, 'TRUCK/BUS': 1, 'TRAIN': 1, 'Size_vehicle_xl': 1, 'VAN': 1,
                    'PICKUP': 1,
                    'MOTORCYCLE': 2, 'RIDER': 2, 'BICYCLE': 2, 'BIKE': 2, 'Two_wheel_without_rider': 2,
                    'Rider': 2,
                    'OTHER_RIDEABLE': 2, 'OTHER-RIDEABLE': 2,
                    'PEDESTRIAN': 3, 'BABY_CARRIAGE': 3, 'SHOPPING-CART': 4, 'OTHER-OBJECT': 4, 'TRAILER': 1
                    }


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


class AiMotiveDataset(tdata.Dataset):
    """
    Multimodal Autonomous Driving dataset.
    The dataset consists of four cameras, two radars, one lidar sensor, and corresponding
    3D bounding box annotations of dynamic objects.

    Attributes:
        dataset_index: a list of keyframe paths
        data_loader: a DataLoader class for loading multimodal sensor data.
    """
    def __init__(self, root_dir: str, pc_range: List[float], split: str = 'train', bda_aug_conf=None, use_cam=True, use_lidar=True,
                 use_radar=True, look_back=0, look_forward=0):
        """
        Args:
            root_dir: path to the dataset
            split: data split, either train or val
        """
        self.split = split
        self.dataset_index = self.get_frames(root_dir, split, look_back, look_forward)
        self.data_loader = DataLoader(self.dataset_index, split, pc_range, use_cam, use_lidar, use_radar, look_back, look_forward)
        self.bda_aug_conf = bda_aug_conf
        self.image_augmentation = A.Compose([
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(0.15, 0.15),
            A.CoarseDropout(max_height=24, max_width=24),
        ])
        self.use_cam   = use_cam
        self.use_lidar = use_lidar

    def __len__(self):
        return len(self.dataset_index)

    def sample_bda_augmentation(self):
        """Generate bev data augmentation values based on bda_config."""
        if self.split == 'train':
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False

        return rotate_bda, scale_bda, flip_dx, flip_dy

    @torch.no_grad()
    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        transform_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (transform_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:9] = (transform_mat[:2, :2] @ gt_boxes[:, 7:9].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, transform_mat

    @torch.no_grad()
    def __getitem__(self, index: int):
        for _ in range(30):
            try:
                out = self.data_loader[self.dataset_index[index]]
                break
            except Exception:
                print(f'Error while loading file {index}')

        images      = list()
        extrinsics  = list()
        sensor2egos = list()
        intrinsics  = list()
        path        = out.annotations.path
        point_cloud = torch.Tensor(out.lidar_data.top_lidar.point_cloud)

        for camera in out.camera_data.items:
            if camera.image is None:
                continue

            camera.image = self.image_augmentation(image=camera.image)['image']
            new_img = torch.Tensor(camera.image).permute(2, 0, 1)

            new_img = torch.cat([new_img, torch.ones(*new_img.shape[1:]).unsqueeze(0) * out.camera_data.timestamp])

            extrinsic = torch.Tensor(camera.camera_params.extrinsic)
            sensor2ego = torch.linalg.inv(extrinsic)
            intrinsic = torch.Tensor(camera.camera_params.intrinsic)

            images.append(new_img)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            sensor2egos.append(sensor2ego)

        images      = torch.stack(images)
        extrinsics  = torch.stack(extrinsics)
        sensor2egos = torch.stack(sensor2egos)
        intrinsics  = torch.stack(intrinsics)
        objects     = out.annotations.objects

        # BEV augmentation
        bda_mat = torch.zeros((4, 4))
        bda_mat[3, 3] = 1
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        out.annotations.objects, bda_rot = self.bev_transform(objects, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)

        point_cloud[:, :3] = point_cloud[:, :3] @ bda_rot.T
        bda_mat[:3, :3] = bda_rot

        return images, point_cloud, extrinsics, sensor2egos, intrinsics, objects, bda_mat, path

    def get_frames(self, path: str, split: str = 'train', look_back=0, look_forward=0) -> List[str]:
        """
        Collects the keyframe paths.

        Args:
            path: path to the dataset
            split: data split, either train or val

        Returns:
            data_paths: a list of keyframe paths

        """
        data_paths = []
        odd_path = os.path.join(path, split)
        for odd in os.listdir(odd_path):
            for seq in os.listdir(os.path.join(odd_path, odd)): # [:100]:
                seq_path = os.path.join(odd_path, odd, seq)
                sequence = Sequence(seq_path, look_back, look_forward)
                data_paths.extend(sequence.get_frames())

        return data_paths


def collate_aim(items: List[Tuple[DataItem, np.ndarray]]):
    images       = list()
    intrinsics   = list()
    sensor2egos  = list()
    all_boxes    = list()
    all_labels   = list()
    paths        = list()
    pointclouds  = list()
    bda_mats     = list()
    extrinsics   = list()

    for image, point_cloud, extrinsic, sensor2ego, intrinsic, objects, bda_mat, path in items:
        image      = image.unsqueeze(0)
        sensor2ego = sensor2ego.unsqueeze(0)
        extrinsic  = extrinsic.unsqueeze(0)
        intrinsic  = intrinsic.unsqueeze(0)

        images.append(image)
        sensor2egos.append(sensor2ego)
        intrinsics.append(intrinsic)
        pointclouds.append(point_cloud)
        paths.append(path)
        bda_mats.append(bda_mat)
        extrinsics.append(extrinsic)

        if len(objects) > 0:
            gt_boxes = objects[:, 0:9]
            gt_labels = objects[:, 9]
            
            all_boxes.append(gt_boxes.float())
            all_labels.append(gt_labels.long())
        else:
            all_boxes.append(torch.zeros((0, 9)))
            all_labels.append(torch.zeros(0, ))

    images       = torch.stack(images)
    sensor2egos  = torch.stack(sensor2egos)
    extrinsics   = torch.stack(extrinsics)
    intrinsics   = torch.stack(intrinsics)
    bda_mats     = torch.stack(bda_mats)
    # pointclouds  = torch.stack(pointclouds)

    mats = {'sensor2ego_mats': sensor2egos,
            'extrinsics': extrinsics,
            'intrin_mats': intrinsics,
            'bda_mat': bda_mats}

    img_metas = [{'path': path, 'box_type_3d': LiDARInstance3DBoxes} for path in paths]

    return images, mats, img_metas, pointclouds, all_boxes, all_labels


if __name__ == '__main__':
    root_dir = Path('/s/home/gabor.nemeth/repos/OpenPCDet/data/custom')
    root_directory = "/h/aimotive_dataset/"
    train_dataset = AiMotiveDataset(root_directory, split='train')
    loader = tdata.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_aim)
    for d in loader:
        print(d)
        a = d

    # for data in tqdm.tqdm(train_dataset.data_loader.data_paths):
    #
    #     idx = int(Path(data).name.split('.')[0].split('_')[-1])
    #     name = f'{data.split("/")[-6]}_{data.split("/")[-5]}_{idx}'

        # lidar = data.lidar_data.top_lidar.point_cloud[..., :4]
        # lidar[:, 3] /= 255.
        # np.snpave(str(lidar_dir / f'{name}.npy'), lidar)
        #
        # obj_lines = [object_to_string(obj) for obj in data.annotations.objects]
        # with open(str(labels_dir / f'{name}.txt'), 'w') as f:
        #     f.writelines(obj_lines)
        #
        # if 'train' in data:
        #     train_indices.append(f'{name}\n')
        #
        # if 'val' in data:
        #     val_indices.append(f'{name}\n')
    #
    # with open(root_dir / 'ImageSets' / 'train.txt', 'w') as f:
    #     f.writelines(train_indices)

    # with open(root_dir / 'ImageSets' / 'val.txt', 'w') as f:
    #     f.writelines(val_indices)