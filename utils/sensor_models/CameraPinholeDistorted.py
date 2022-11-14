from utils.sensor_models.CameraBase import CameraBase, TensorLike, isTensor, isNumpy
from utils.sensor_models.CameraPinhole import CameraPinhole
from typing import Union, Tuple, Callable
from functools import partial
from scipy.spatial.transform import Rotation

import torch.nn.functional
import cv2
import numpy as np
import torch
import json

class CameraPinholeDistorted(CameraPinhole):

    model_name = 'distorted_pinhole'

    def __init__(self, intrinsic:np.ndarray, distortion_coeffs:Union[list, tuple], image_size:Union[list, tuple],
                 rotation:Rotation, translation:Union[list,np.ndarray]):

        super().__init__(intrinsic, image_size, rotation, translation)
        self.model_name = CameraPinholeDistorted.model_name
        assert len(distortion_coeffs) == 5, 'Length of coeffs list must be 5 [k1,k2,p1,p2,k3]'
        self.dist_coeffs = distortion_coeffs

    def image2ray(self, x: TensorLike, channel_first: bool=False) -> TensorLike:

        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        distorted_rays = CameraPinhole.image2ray(self, x, True)

        k1,k2,p1,p2,k3 = self.dist_coeffs
        n_iterations = 20
        for i in range(n_iterations):
            xx = distorted_rays[0] * distorted_rays[0]
            yy = distorted_rays[1] * distorted_rays[1]

            r2 = xx + yy
            _2xy = 2.0 * distorted_rays[0] * distorted_rays[1]

            radial_distortion = 1.0 + (k1 + (k2 + k3 * r2) * r2) * r2
            tangential_distortion_X = np.array(p1 * _2xy + p2 * (r2 + 2.0 * xx))
            tangential_distortion_Y = np.array(p1 * (r2 + 2.0 * yy) + p2 * _2xy)

            distorted_rays[0] = (distorted_rays[0] - tangential_distortion_X) / radial_distortion
            distorted_rays[1] = (distorted_rays[1] - tangential_distortion_Y) / radial_distortion

        rays = distorted_rays

        if not channel_first:
            rays = torch.movedim(rays, 0, -1) if isTensor(x) else np.moveaxis(rays, 0, -1)

        return rays

    def ray2image(self, x: TensorLike, channel_first: bool=False) -> Tuple[TensorLike, TensorLike]:

        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        x = x/x[2:3]
        x_coord = x[0]
        y_coord = x[1]

        k1,k2,p1,p2,k3 = self.dist_coeffs

        r2 = x_coord * x_coord + y_coord * y_coord
        r4 = r2 * r2
        r6 = r4 * r2

        coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        distorted_rays = torch.ones_like(x, dtype=torch.float32, device=x.device) if isTensor(x) else np.ones_like(x, dtype=np.float32)

        distorted_rays[0] = (x_coord * coefficient + 2.0 * p1 * x_coord * y_coord + p2 * (r2 + 2.0 * x_coord * x_coord))
        distorted_rays[1] = (y_coord * coefficient + 2.0 * p2 * x_coord * y_coord + p1 * (r2 + 2.0 * y_coord * y_coord))

        x_image, invalid_projection_mask = CameraPinhole.ray2image(self, distorted_rays, True)

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
        img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0006395.jpg')
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
        target_intrinsic[1, 2] = 128 #55
        target_img_size = [256, 1024]
        start = time.time()
        target_cam = CameraPinholeDistorted(target_intrinsic, [0.1,0.1,0.1,0.1,0.0], target_img_size, rotation=target_rotation, translation=[0, 0, 0])
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
