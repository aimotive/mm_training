import json
import os
from typing import Dict

import numpy as np

RADAR_MAPPING = {'F_LRR_C': 'F_LRR_C',
                 'F_LRR_C_FW4': 'F_LRR_C',
                 'B_LRR_C': 'B_LRR_C',
                 'B_SRR_R': 'B_SRR_R',
                 'B_SRR_L': 'B_SRR_L',
                 'F_SRR_R': 'F_SRR_R',
                 'F_SRR_L': 'F_SRR_L',}


class RadarData:
    """ Stores the RadaraDataItem for each radar.

    Attributes
        front_radar: raw data, point cloud, and extrinsic matrix for the front radar
        back_radar: raw data, point cloud, and extrinsic matrix for the back radar
    """

    def __init__(self, front_raw, front_pcd: np.array, back_raw, back_pcd: np.array,
                 extrinsic_by_sensor: Dict[str, np.array]):
        """
        Args:
            front_raw: radar targets for front radar as a dict with azimuth, elevation, range, speed, rcs, power, noise attributes
            front_pcd: point cloud for front radar, numpy array (shape: [N, 5]), format: [x, y, z, speed, power]
            back_raw: radar targets for back radar as a dict with azimuth, elevation, range, speed, rcs, power, noise attributes
            back_pcd: point cloud for back radar, numpy array (shape: [N, 5]), format: [x, y, z, speed, power]
            extrinsic_by_sensor: dict, key: sensor name, value: extrinsic matrix
        """
        self.front_radar = RadarDataItem('front_radar', front_raw, front_pcd, extrinsic_by_sensor['F_LRR_C'])
        self.back_radar = RadarDataItem('back_radar', back_raw, back_pcd, extrinsic_by_sensor['B_LRR_C'])

        self.index = 0
        self.items = [self.front_radar, self.back_radar]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.items):
            self.index = 0
            raise StopIteration

        item = self.items[self.index]
        self.index += 1

        return item


class RadarDataItem:
    """
    Stores the image and camera parameters for a camera with a given frame id.
    Coordinate system: x -> forward, y -> left, z -> top
    x, y, z coordinates are stored in meters.

    Attributes
        name: sensor name
        raw_data: radar targets as a dict with azimuth, elevation, range, speed, rcs, power, noise attributes
        point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, speed, power]
        extrinsic: body to sensor matrix, shape: [4, 4]
    """

    def __init__(self, name: str, raw_data: dict, point_cloud: np.array, extrinsic: np.array):
        """
        Args:
            name: sensor name
            raw_data: keys: id, targets list
                      target list elements are dict with keys: azimuth, elevation, range, speed, rcs, power, noise
            point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, speed, power]
            extrinsic: body to sensor matrix, shape: [4, 4]
        """
        self.name = name
        self.raw_data = raw_data
        self.point_cloud = point_cloud
        self.extrinsic = extrinsic


def load_radar_data(data_folder: str, frame_id: str) -> RadarData:
    """
    Loads data for each radar with a given frame id.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        radar_data: a RadarData instance with front and back radar (raw data, point cloud, and extrinsic matrix)
    """
    front_radar_path = os.path.join(data_folder, 'sensor', 'radar', 'F_LRR_C', 'F_LRR_C_' + frame_id + '.json')
    back_radar_path = os.path.join(data_folder, 'sensor', 'radar', 'B_LRR_C', 'B_LRR_C_' + frame_id + '.json')
    cali_path = os.path.join(data_folder, 'sensor', 'calibration')
    extrinsic_by_sensor = read_calibrations(cali_path)

    with open(front_radar_path, 'r') as front_radar_file:
        front_raw = json.load(front_radar_file)
        front_pcd = radar_json_to_pcd(front_raw, extrinsic_by_sensor['F_LRR_C'])
    with open(back_radar_path, 'r') as back_radar_file:
        back_raw = json.load(back_radar_file)
        back_pcd = radar_json_to_pcd(back_raw, extrinsic_by_sensor['B_LRR_C'])

    radar_data = RadarData(front_raw, front_pcd, back_raw, back_pcd, extrinsic_by_sensor)

    return radar_data


def radar_json_to_pcd(json_data: Dict, extrinsic: np.array) -> np.array:
    """
    Converts radar dict to numpy array, and from sensor to body coordinates.

    Args:
        json_data: stores the radar targets, radar target attributes: azimuth, elevation, range, speed, rcs, power, noise
        extrinsic: body to sensor transformation matrix

    Returns:
        pcd: point cloud, shape: [N, 5], attributes: x, y, z, speed, power
    """
    targets = json_data['targets']

    pcd = np.ndarray(shape=(len(targets), 5), dtype=np.float32)
    for i, target in enumerate(targets):
        elevation = target['elevation']
        r = target['range']
        azimuth = target['azimuth']
        speed = target['speed']
        power = target['power']

        position = np.array([
            [r * np.cos(elevation) * np.cos(azimuth)],
            [r * np.cos(elevation) * np.sin(azimuth)],
            [r * np.sin(elevation)],
            [1]])

        # Calibration matrix is body to sensor, we need to invert it.
        position = np.linalg.inv(extrinsic) @ position

        pcd[i, 0:3] = position[0:3, 0]
        pcd[i, 3] = speed
        pcd[i, 4] = power

    return pcd


def read_calibrations(path: str) -> Dict[str, np.array]:
    """
    Reads radar extrinsic calibrations from json files.

    Args:
        path: path to the directory of calibration file

    Raises:
        JSONDecodeError: error during camera parameters file loading

    Returns:
        extrinsic_by_sensor: dict, key: sensor name, value: extrinsic matrix
    """
    cali_json_path = os.path.join(path, 'calibration.json')
    if os.path.exists(cali_json_path):
        with open(cali_json_path, 'r') as stream:
            try:
                json_data = json.load(stream)
            except json.decoder.JSONDecodeError:
                print(f"Failed to load: {cali_json_path}")

        extrinsic_by_sensor = {}
        for sensor,v in json_data.items():
            if 'LRR' in sensor and 'RT_sensor_from_body' in v:
                extrinsic_by_sensor[RADAR_MAPPING[sensor]] = np.array(v[f'RT_sensor_from_body'])

    return extrinsic_by_sensor
