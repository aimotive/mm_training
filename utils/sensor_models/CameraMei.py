from utils.sensor_models.CameraBase import CameraBase, TensorLike, isTensor, isNumpy
from utils.sensor_models.CameraPinholeDistorted import CameraPinholeDistorted
from typing import Union, Tuple, Callable
from functools import partial
from scipy.spatial.transform import Rotation

import torch.nn.functional
import cv2
import numpy as np
import torch
import json

class CameraMei(CameraPinholeDistorted):

    model_name = 'mei'

    def __init__(self, intrinsic: np.ndarray, xi:float, distortion_coeffs: Union[list, tuple], image_size:Union[list,tuple],
                 rotation:Rotation, translation:Union[np.ndarray,list]):
        super().__init__(intrinsic, distortion_coeffs, image_size, rotation, translation)
        self.model_name = CameraMei.model_name
        self.xi = xi

    def image2ray(self, x: TensorLike, channel_first: bool=False) -> TensorLike:
        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        undist_rays = CameraPinholeDistorted.image2ray(self, x, True)
        # project to unit sphere - OpenCV Omnidir implementation
        r2 = undist_rays[0] * undist_rays[0] + undist_rays[1] * undist_rays[1]
        a = (r2 + 1.0)
        b = 2 * self.xi * r2
        cc = r2 * self.xi * self.xi - 1
        Zs = (-b + np.sqrt(b * b - 4 * a * cc)) / (2 * a)
        ret_x = undist_rays[0] * (Zs + self.xi)
        ret_y = undist_rays[1] * (Zs + self.xi)
        ret_z = Zs

        rays = torch.stack([ret_x, ret_y, ret_z], dim=0) if isTensor(x) else np.stack([ret_x, ret_y, ret_z], axis=0)

        if not channel_first:
            rays = torch.movedim(rays, 0, -1) if isTensor(x) else np.moveaxis(rays, 0, -1)

        return rays

    def ray2image(self, x: TensorLike, channel_first: bool=False) -> Tuple[TensorLike, TensorLike]:
        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        # project to unit sphere
        norm = torch.norm(x, None, 0, True) if isTensor(x) else np.linalg.norm(x, None, 0, True)
        x[0:1] = x[0:1] / norm
        x[1:2] = x[1:2] / norm
        x[2:3] = x[2:3] / norm + self.xi
        to_clip = x[2:3] < 1e-5
        # x[2:3][to_clip] = 1e-5 * torch.sign(x[2:3]) if isTensor(x) else np.sign(x[2:,3])
        x[2:3] = np.where(to_clip, 1e-5 * torch.sign(x[2:3]) if isTensor(x) else np.sign(x[2:3]), x[2:3])
        # distorted projection to plane
        x_image, invalid_projection_mask = CameraPinholeDistorted.ray2image(self, x, True)

        if not channel_first:
            x_image = torch.movedim(x_image, 0, -1) if isTensor(x) else np.moveaxis(x_image, 0, -1)
            invalid_projection_mask = torch.movedim(invalid_projection_mask, 0, -1) if isTensor(x) else np.moveaxis(invalid_projection_mask, 0, -1)

        return x_image, invalid_projection_mask

    '''
    Parent class handles the rest
    '''

if __name__ == '__main__':

    import time
    from scipy.spatial.transform import Rotation
    from CameraPinhole import CameraPinhole

    extrinsic_RT = np.eye(4)
    intrinsic_mat = np.array([[1050, 0, 512],
                              [0, 1050, 84],
                              [0, 0, 1]])
    img_size = [1024, 269]

    r = Rotation.from_euler('yxz', [0, 0, 0], True)
    camera = CameraPinhole(intrinsic=intrinsic_mat, image_size=img_size, rotation=r, translation=[0, 0, 0])
    # region check conversion
    runtimes = []
    for ry in range(-30, 30, 1):
        img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0054763.jpg')
        img_og = np.flipud(np.transpose(img_og, [1, 0, 2]))

        # channel last
        # ch_first = False
        # img = img_og

        # channel first
        img = img_og.transpose([2, 0, 1])
        ch_first = True

        # tensor conversion
        # img = torch.from_numpy(img.copy())

        source_rotation = Rotation.from_euler('yxz', [0, 0, 0], True)
        source_intrinsic = intrinsic_mat
        source_intrinsic[1, 2] = 390
        source_cam = CameraPinhole(source_intrinsic, img_og.shape[0:2], rotation=source_rotation, translation=[0, 0, 0])

        target_rotation = Rotation.from_euler('yxz', [0, ry, 0], degrees=True)
        target_intrinsic = intrinsic_mat
        target_intrinsic[1, 2] = 128  # 55
        target_img_size = [256, 1024]
        start = time.time()
        target_cam = CameraMei(target_intrinsic, 5.0, [0.1, 0.1, 0.0, 0.0, 0.0], target_img_size, rotation=target_rotation, translation=[0, 0, 0])
        target_img = target_cam.convert_from_cam(img, source_cam, channel_first=ch_first, store_cvtFunction=True)
        end = time.time()
        runtimes.append(end - start)

        if isTensor(target_img):
            target_img = target_img.numpy()

        if ch_first:
            target_img = target_img.transpose([1, 2, 0])

        cv2.imshow("Target crop", target_img)
        cv2.waitKey(10)

    runtimes = np.asarray(runtimes)
    print(f'avg runtimes : {runtimes.mean()}')
    print(target_cam.save_to_dict())




