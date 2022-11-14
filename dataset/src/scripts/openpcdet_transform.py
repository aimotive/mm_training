import argparse
import multiprocessing
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from src.aimotive_dataset import AiMotiveDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Example script for visualizing aiMotive Multimodal Dataset.')

    parser.add_argument("--root_dir", required=True,
                        type=Path, help="Root dir of aiMotive Multimodal Dataset.")
    parser.add_argument("--out_dir", required=True,
                        type=Path, help="Output directory to save transformed lidar point cloud.")
    return parser.parse_args()


def object_to_string(object):
    x = object['BoundingBox3D Origin X']
    y = object['BoundingBox3D Origin Y']
    z = object['BoundingBox3D Origin Z']

    dx = object['BoundingBox3D Extent X']
    dy = object['BoundingBox3D Extent Y']
    dz = object['BoundingBox3D Extent Z']
    ori = Rotation.from_quat((object['BoundingBox3D Orientation Quat X'],
                                     object['BoundingBox3D Orientation Quat Y'],
                                     object['BoundingBox3D Orientation Quat Z'],
                                     object['BoundingBox3D Orientation Quat W'])).as_euler('xyz', degrees=False)[2]


    category_name = object['ObjectType']
    return f'{x} {y} {z} {dx} {dy} {dz} {ori} {category_name}\n'


def save_files(i):
    path = dataset.data_loader.data_paths[i]
    idx = int(Path(path).name.split('.')[0].split('_')[-1])
    name = f'{path.split("/")[-6]}_{path.split("/")[-5]}_{idx}'

    data = dataset[i]
    lidar = data.lidar_data.top_lidar.point_cloud[..., :4]
    lidar[:, 3] /= 255.

    # stack input to x, y, z, type, intensity, power, speed
    lidar = np.hstack([lidar[:, 0:3], np.zeros((lidar.shape[0], 1)), np.expand_dims(lidar[:, 3], -1), np.zeros((lidar.shape[0], 2))])
    radar = np.vstack([data.radar_data.back_radar.point_cloud, data.radar_data.front_radar.point_cloud])
    radar = np.hstack([radar[:, 0:3], np.ones((radar.shape[0], 1)), np.zeros((radar.shape[0], 1)), radar[:, 3:]])
    radar_lidar = np.vstack([radar, lidar]).astype(np.float32)

    np.save(str(lidar_dir / f'{name}.npy'), radar_lidar)

    obj_lines = [object_to_string(obj) for obj in data.annotations.objects]
    with open(str(labels_dir / f'{name}.txt'), 'w') as f:
        f.writelines(obj_lines)

    return name


if __name__ == '__main__':
    args = parse_args()
    root_directory = args.root_dir
    out_dir = args.out_dir

    lidar_dir = out_dir / 'points'
    labels_dir = out_dir / 'labels'
    imageset_dir = out_dir / 'ImageSets'

    lidar_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    imageset_dir.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val']
    for split in splits:
        dataset = AiMotiveDataset(root_directory, split=split)

        names = list()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for name in tqdm(pool.imap_unordered(save_files, range(len(dataset)))):
                names.append(name)

        with open(imageset_dir / f'{split}.txt', 'w') as f:
            f.writelines(names)
