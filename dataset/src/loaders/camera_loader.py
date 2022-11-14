import json
import os
from typing import Dict, Tuple, List

import cv2
import numpy as np

from dataset.src.camera_params import CameraParams

CAMERA_MAPPING = {
    'FrontCenter': 'F_STEREO_L',
    'F_STEREO_L': 'F_STEREO_L',
    'F_MIDLONGRANGECAM_CL': 'F_STEREO_L',
    'B_MIDRANGECAM_C': 'B_MIDRANGECAM_C',
    'M_FISHEYE_L': 'M_FISHEYE_L',
    'M_FISHEYE_R': 'M_FISHEYE_R',
}


class CameraData:
    """ Stores the CameraDataItem for each camera.

    Attributes
        front_camera: image and camera parameters for the front camera.
        back_camera: image and camera parameters for the back camera.
        left_camera: image and camera parameters for the left camera.
        right_camera: image and camera parameters for the right camera.
    """

    def __init__(self, front_img: np.array, back_img: np.array, left_img: np.array,
                 right_img: np.array, camera_params_by_sensor: Dict[str, CameraParams], timestamp: int):
        """
        Args:
            front_img: front image as numpy array
            back_img: back image as numpy array
            left_img: left image as numpy array
            right_img: right image as numpy array
            camera_params_by_sensor: dict, key: sensor name, value: CameraParams
        """
        self.front_camera = CameraDataItem('front_cam', front_img, camera_params_by_sensor['F_STEREO_L'])
        self.back_camera = CameraDataItem('back_cam', back_img, camera_params_by_sensor['B_MIDRANGECAM_C'])
        self.left_camera = CameraDataItem('left_cam', left_img, camera_params_by_sensor['M_FISHEYE_L'])
        self.right_camera = CameraDataItem('right_cam', right_img, camera_params_by_sensor['M_FISHEYE_R'])
        self.timestamp = timestamp
        self.index = 0
        self.items = [self.front_camera, self.back_camera, self.left_camera, self.right_camera]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.items):
            self.index = 0
            raise StopIteration

        item = self.items[self.index]
        self.index += 1

        return item


class CameraDataItem:
    """
    Stores the image and camera parameters for a camera with a given frame id.

    Attributes
        name: camera name
        image: images stored as a numpy array
        camera_params: camera parameters for the given data item
    """

    def __init__(self, name: str, image: np.array, camera_params: CameraParams):
        """
        Args:
            name: camera name
            image: images stored as a numpy array
            camera_params: camera parameters for the given data item
        """
        self.name = name
        self.image = image
        self.camera_params = camera_params


def read_timestamp(data_folder, frame_id):
    timestamp_path = os.path.join(data_folder, 'sensor', 'camera', 'sync_frame2host.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)

    return timestamps[str(int(frame_id))]


def load_camera_data(data_folder: str, frame_id: str, use_cam: bool) -> CameraData:
    """
    Loads data for each camera with a given frame id.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        camera_data: a CameraData instance with front, back, left, right image and camera parameters.
    """

    front_cam_path, back_cam_path, left_cam_path, right_cam_path = get_camera_paths(data_folder, frame_id)
    cali_path = os.path.join(data_folder, 'sensor', 'calibration')
    timestamp = read_timestamp(data_folder, frame_id)

    camera_params_by_sensor = read_camera_params(cali_path)

    if use_cam:
        front_img, back_img = cv2.imread(front_cam_path), cv2.imread(back_cam_path)
    else:
        front_img, back_img = cv2.imread(front_cam_path), None
    # left_img, right_img = cv2.imread(left_cam_path), cv2.imread(right_cam_path)
    left_img, right_img = None, None
    camera_data = CameraData(front_img, back_img, left_img, right_img, camera_params_by_sensor, timestamp)

    return camera_data


def get_camera_paths(data_folder: str, frame_id: str) -> Tuple[str, str, str, str]:
    """
    Collects the path of the image with a given frame id for each camera.

    Args:
        data_folder: the path of the sequence from where data is loaded
        frame_id: id if the loadable camera data, e.g. 0033532

    Returns:
        a tuple with the path of the front, back, left, and right camera, respectively
    """
    cam_base_path = os.path.join(data_folder, 'sensor', 'camera')
    cam_names = os.listdir(cam_base_path)
    front_cam_name = [cam_name for cam_name in cam_names if cam_name[0] == 'F' and cam_name[-1] == 'L'][0]
    front_cam_path = os.path.join(cam_base_path, front_cam_name, front_cam_name + "_" + frame_id + '.jpg')
    back_cam_path = os.path.join(cam_base_path, 'B_MIDRANGECAM_C', 'B_MIDRANGECAM_C_' + frame_id + '.jpg')
    left_cam_path = os.path.join(cam_base_path, 'M_FISHEYE_L', 'M_FISHEYE_L_' + frame_id + '.jpg')
    right_cam_path = os.path.join(cam_base_path, 'M_FISHEYE_R', 'M_FISHEYE_R_' + frame_id + '.jpg')

    return front_cam_path, back_cam_path, left_cam_path, right_cam_path


def read_camera_params(path: str) -> Dict[str, Dict]:
    """
    Reads camera parameters from the calibration json file

    Args:
        path: the path of the folder where calibration file is stored

    Raises:
        JSONDecodeError: error during camera parameters file loading

    Returns:
        a dict with camera names as keys and a dict with keys intrinsic, extrinsic, dist, model, and xi.
    """
    cali_json_path = os.path.join(path, 'calibration.json')
    if os.path.exists(cali_json_path):
        with open(cali_json_path, 'r') as stream:
            try:
                camera_params_json = json.load(stream)
            except json.decoder.JSONDecodeError:
                print(f"Failed to load: {cali_json_path}")

        camera_params_by_sensor = {}
        for sensor, params in camera_params_json.items():
            if sensor in CAMERA_MAPPING and 'RT_sensor_from_body' in params:
                model = params['model']
                intrinsic, _ = get_intrinsics(params['focal_length_px'], params['principal_point_px'])
                extrinsic = np.array(params[f'RT_sensor_from_body'])
                dist = np.array(params['distortion_coeffs']) if 'distortion_coeffs' in params else np.array([0., 0., 0., 0., 0.])
                camera_params = CameraParams(intrinsic, extrinsic, dist, model)
                if 'FISHEYE' in sensor and model == 'mei':
                    camera_params.xi = params['xi']
                camera_params_by_sensor[CAMERA_MAPPING[sensor]] = camera_params

    return camera_params_by_sensor


def get_intrinsics(focal_length: List[float], principal_point: List[float]) -> Tuple[np.array, np.array]:
    """

    Args:
        focal_length: list of focal length [f_x, f_y]
        principal_point: list of principal point [p_x, p_y]

    Returns:
        ray to image (i.e. intrinsic, shape: [3, 4]) and image to ray (shape: [3, 3] matrices
    """
    f = focal_length
    p = principal_point

    ray_to_image = np.array([
        [f[0], 0   , p[0], 0],
        [0   , f[1], p[1], 0],
        [0   , 0   , 1   , 0]
    ])

    image_to_ray = np.array([
        [1 / f[0], 0       , -p[0] / f[0]],
        [0       , 1 / f[1], -p[1] / f[1]],
        [0       , 0       ,  1          ]
    ])

    return ray_to_image, image_to_ray



