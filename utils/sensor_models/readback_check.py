import os

from CameraBase import CameraBase
from CameraPinhole import CameraPinhole
from CameraPinholeDistorted import CameraPinholeDistorted
from CameraMei import CameraMei
from CameraEquirect import CameraEquirect

from scipy.spatial.transform import Rotation

import numpy as np
import json

if __name__ == '__main__':

    extrinsic_RT = np.eye(4)
    intrinsic_mat = np.array([[1050, 0, 512],
                              [0, 1050, 84],
                              [0, 0, 1]])
    img_size = [1024, 269]

    r = Rotation.from_euler('yxz', [0, 0, 0], True)
    camera_pinhole = CameraPinhole(intrinsic=intrinsic_mat, image_size=img_size, rotation=r, translation=[0, 0, 0])
    camera_pinhole_dist = CameraPinholeDistorted(intrinsic=intrinsic_mat, distortion_coeffs=[0.1,0.1,0.0,0.2,0.01], image_size=img_size, rotation=r, translation=None)
    camera_mei = CameraMei(intrinsic=intrinsic_mat, xi=0.3, distortion_coeffs=[0.1,0.1,0.0,0.2,0.01], image_size=img_size, rotation=r, translation=None)
    camera_equirect = CameraEquirect(horizontal_fov_limits_deg=[-30,30], vertical_fov_limits_deg=[-15,25], image_size=img_size, rotation=r, translation=None)

    cameras = [camera_pinhole,camera_pinhole_dist, camera_mei, camera_equirect]

    for cam in cameras:
        s = cam.save_to_string()
        cam_readback = cam.load_from_dict(json.loads(s))
        print(type(cam), type(cam_readback), cam.save_to_string() == cam_readback.save_to_string())
        # print(cam_readback.__repr__())
        # cam.save_to_json(f'./{cam.model_name}_test_cali.json')

    # from src.imgproc.sensor_models import make_from_json
    #
    # files = os.listdir('.')
    # json_files = [n for n in files if os.path.isfile(n) and n.endswith('.json')]
    # print(json_files)
    #
    # for json_file in json_files:
    #     cam = make_from_json(os.path.join('.',json_file))
    #     print(json_file)
    #     print(cam)

