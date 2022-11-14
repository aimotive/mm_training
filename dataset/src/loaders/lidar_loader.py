import json
import os

import laspy
import numpy as np


class LidarData:
    """ Stores the LidarDataItem for each lidar.

    Attributes
        top_lidar: LidarDataItem for top lidar
    """

    def __init__(self, point_cloud: np.array):
        """
        Args:
            point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
        """
        self.top_lidar = LidarDataItem('top_lidar', point_cloud)


class LidarDataItem:
    """
    Stores the point cloud for a lidar with a given frame id.
    Coordinate system: x -> forward, y -> left, z -> top
    x, y, z coordinates are stored in meters.

    Attributes
        name: sensor name
        point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
    """

    def __init__(self, name: str, point_cloud: np.array):
        """
        Args:
            name: sensor name
            point_cloud: numpy array (shape: [N, 5]), format: [x, y, z, intensity, timestamp]
        """
        self.name = name
        self.point_cloud = point_cloud


def load_lidar_data(data_folder: str, frame_id: str, look_back: int = 0, look_forward: int = 0) -> LidarData:
    """
    Loads data for each lidar with a given frame id. Current dataset has only one lidar.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        lidar_data: a LidarData instance with a top lidar point cloud.
    """
    # lidar_path = os.path.join(data_folder, 'dynamic', 'raw-revolutions', 'frame_' + frame_id + '.laz')
    egomotion_path = os.path.join(data_folder, 'sensor', 'gnssins', 'egomotion.json')

    with open(egomotion_path) as f:
        egomotion = json.load(f)

    RT_main_frame = np.array(egomotion[str(int(frame_id))]).reshape(4, 4)

    lidar_frames = list()
    for frame in range(int(frame_id) - look_back, int(frame_id) + look_forward + 1):
        current_lidar_path = os.path.join(data_folder, 'dynamic', 'raw-revolutions', 'frame_' + str(frame).zfill(7) + '.laz')

        RT_current  = np.array(egomotion[str(frame)]).reshape(4, 4)
        RT_transform = np.linalg.inv(RT_main_frame) @ RT_current

        current_lidar_data = read_lidar(current_lidar_path)
        current_lidar_data = filter_ego_car(current_lidar_data)
        current_lidar_data_coords = np.hstack([current_lidar_data[:, :3], np.ones((current_lidar_data.shape[0], 1))])
        current_lidar_data[:, :3] = (current_lidar_data_coords @ RT_transform.T)[:, :3]
        lidar_frames.append(current_lidar_data)

    return LidarData(np.concatenate(lidar_frames))


def filter_ego_car(pc):
    filter_x = np.logical_and(pc[:, 0] < 3.8, pc[:, 0] > -1.2)
    filter_y = np.logical_and(pc[:, 1] < 1.7, pc[:, 1] > -1.7)
    valid = ~np.logical_and(filter_x, filter_y)
    return pc[valid]


def read_lidar(lidar_path):
    with laspy.open(lidar_path) as fh:
        las = fh.read()
        lidar_pcd = np.array([las.x, las.y, las.z, las.intensity, las.gps_time], dtype=np.float32)
        lidar_pcd = lidar_pcd.T
    return lidar_pcd
