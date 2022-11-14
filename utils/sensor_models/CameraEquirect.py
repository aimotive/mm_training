import numpy as np
from functools import partial
import torch
from typing import Tuple, Callable, Union, List
from scipy.spatial.transform import Rotation

from utils.sensor_models.CameraBase import CameraBase, isTensor, TensorLike
from utils.sensor_models.CameraPinhole import CameraPinhole

class CameraEquirect(CameraBase):

    '''
        Equirectangular camera.

        parameters:
            - horizontal_fov_limits_deg : [int,int] lower and upper horizontal angular limit of FOV [left, right]
            - vertical_fov_limits_deg : [int,int] lower and upper vertical angular limit of FOV [nose down, nose up]
            - image_size : [Vertical Size, Horizontal Size]
            Extrinsics arguments:
            - rotation : rotation component in general scipy.spatial.transform.Rotation
                         It describes the orientation of the camera relative to the world/rig/body
            - translation: translation component
                         It desctibes the orientation of the camera relative to the world/rig/body

            Both extrinsic and its inverse gets computed from this.

            Note: in the camera we use Z-forward frame and the axis swap, if any, must be in the rotation part.
    '''

    model_name = 'equirect'

    def __init__(self, horizontal_fov_limits_deg: Union[Tuple[float,float],List[float]], vertical_fov_limits_deg: Union[Tuple[float,float],List[float]],
                 image_size: Union[tuple,list], rotation: Rotation = None, translation: Union[list, np.ndarray] = None):
        super().__init__(rotation, translation)
        self.horizontal_fov_limits_deg = horizontal_fov_limits_deg
        self.vertical_fov_limits_deg = vertical_fov_limits_deg
        self.image_size = image_size
        self.model_name = CameraEquirect.model_name

    def image2ray(self, x: TensorLike, channel_first: bool = False) -> TensorLike:
        '''
            See CameraBase.image2ray.

            x : input coordinates to project (like the image meshgrid)
            channel_first : Whether the input is [2, ...] or [..., 2]. Output will be the same order.

            returns : array or tensor according to arguments. Same shape as 'x'.
        '''

        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        pi = np.pi
        lib = torch if isTensor(x) else np

        theta, phi = self._pixel_to_degree(vertical_px=x[1], horizontal_px=x[0])

        theta *= pi / 180
        phi *= pi / 180

        x = lib.sin(phi) * lib.cos(theta)
        y = lib.sin(theta)
        z = lib.cos(-phi) * lib.cos(theta)

        rays = torch.stack([x, y, z], dim=0) if isTensor(x) else np.stack([x, y, z], axis=0)

        if not channel_first:
            rays = torch.movedim(rays, 0, -1) if isTensor(x) else np.moveaxis(rays, 0, -1)

        return rays

    def ray2image(self, x: TensorLike, channel_first=False) -> Tuple[TensorLike, TensorLike]:
        '''
            See CameraBase.ray2image

            x: The points to project onto the image plane.
            channel_first : Whether 'X' is [3, ...] or [..., 3]

            return:
                x_image : array or tensor of projected coordinates
                invalid_projection_mask : array. Where the projection failed or otherwise became invalid
        '''

        if not channel_first:
            size = len(x.shape)
            perm = np.roll(np.arange(size), shift=1)
            x = x.permute(*perm) if isTensor(x) else np.transpose(x, perm)

        assert x.shape[0] == 3, "Wrong input for CameraEquirect.ray2image"

        lib = torch if isTensor(x) else np
        pi = np.pi

        x, y, z = x

        # South Pole and North Pole are not in our model
        invalid_projection_mask = np.expand_dims(np.logical_and(x == 0, z == 0), -1)

        r = lib.sqrt(x * x + y * y + z * z)
        theta = lib.zeros_like(x)
        phi = lib.zeros_like(x)

        theta = lib.where(r != 0, lib.arcsin(-y / r), theta)

        phi = lib.where(z > 0, lib.arctan(x / z), phi)
        phi = lib.where((z < 0) & (x <= 0), lib.arctan(x / z) - pi, phi)
        phi = lib.where((z < 0) & (x > 0), lib.arctan(x / z) + pi, phi)
        phi = lib.where((z == 0) & (x != 0), pi / 2, phi)

        image = lib.stack([theta, phi]) * 180 / pi

        if not channel_first:
            size = len(image.shape)
            inverse_perm = np.roll(np.arange(size), shift=-1)
            image = image.permute(*inverse_perm) if isTensor(image) else np.transpose(image, inverse_perm)

        return image, invalid_projection_mask

    def get_converter_function(self, source_cam: CameraBase, store_cvtFunction:bool = False) -> Callable[[TensorLike, bool], TensorLike]:
        '''
            Computes a function that can map FROM the SOURCE camera to *THIS* (TARGET) camera.
            source_cam : Source camera.
            store_cvtFunction : Whether to cache the resulting function. If set to TRUE the next time we want to compute a mapping function
                                between two cameras that use the exact same parameters it will skip the computation and
                                return with the precomputed function. The cache is not part of the instance but the class!
                                Use it if you know you will probably remap between a few sets of cameras.
                                WARNING : There is a potential, that some special cases the function table can grow quite large.
        '''

        self.apply_extrinsic_inheritance(source_cam)

        if (self, source_cam) in CameraBase.cvtFunctionTable:
            return CameraBase.cvtFunctionTable[(self, source_cam)]

        # TODO: make it work for yaw augmentations, i.e. when the extrinsic parameters are the same except a horizontal rotation
        # crop is possible
        if (isinstance(source_cam, CameraEquirect) and np.all(source_cam.RT_body_cam == self.RT_body_cam) and
                (self.vertical_fov_limits_deg[1] - self.vertical_fov_limits_deg[0]) / self.image_size[0] ==
                (source_cam.vertical_fov_limits_deg[1] - source_cam.vertical_fov_limits_deg[0]) / source_cam.image_size[0] and
                (self.horizontal_fov_limits_deg[1] - self.horizontal_fov_limits_deg[0]) / self.image_size[1] ==
                (source_cam.horizontal_fov_limits_deg[1] - source_cam.horizontal_fov_limits_deg[0]) / source_cam.image_size[1]):

            min_px = [int(px) for px in self._degree_to_pixel(*source_cam._pixel_to_degree(0, 0))]
            vertical_px = (min_px[0], min_px[0] + source_cam.image_size[0])
            horizontal_px = (min_px[1], min_px[1] + source_cam.image_size[1])
            TL_target = np.array([max(0, vertical_px[0]), max(0, horizontal_px[0])], dtype=np.float32)
            TL_source = (source_cam._degree_to_pixel(*self._pixel_to_degree(*TL_target)))
            BR_target = np.array([min(self.image_size[0], vertical_px[1]), max(self.image_size[1], horizontal_px[1])], dtype=np.float32)
            H_intersection, W_intersection = BR_target - TL_target

            if H_intersection <= 0 or W_intersection <= 0:
                cvtFunction = lambda x, channel_first: np.zeros([x.shape[0], *self.image_size], dtype=np.float32) if channel_first else np.zeros(
                    [*self.image_size, x.shape[-1]], dtype=np.float32)
            else:
                cvtFunction = partial(self.crop_convert, TL_target=TL_target, TL_source=TL_source, H=H_intersection, W=W_intersection)

        else:
            grid_rays = self.grid2ray(channel_first=False)
            if not np.all(self.RT_body_cam == source_cam.RT_body_cam):
                assert np.all(self.RT_body_cam[:3, 3] == source_cam.RT_body_cam[:3, 3]), \
                    "Camera conversion is invalid. The translation of the extrinsic between SOURCE and TARGET camera must match!"
                extrinsic_transform = source_cam.RT_cam_body @ self.RT_body_cam
                grid_rays = (extrinsic_transform @ CameraPinhole.make_homogen(grid_rays.transpose([2, 0, 1])).reshape([4, -1]))[:3].transpose(
                    [1, 0]).reshape(grid_rays.shape)
            grid_mapping_other, validity_mask = source_cam.ray2image(grid_rays, channel_first=False)

            cvtFunction = partial(CameraBase.resample, mapping=grid_mapping_other, invalid_projection_mask=validity_mask)

        return cvtFunction

    def grid2ray(self, tensor_out: bool=False, channel_first: bool=False) -> TensorLike:
        '''
            Utility function to project the image mesh grid.
            tensor_out : Whether we want the results in torch.Tensor or np.ndarray
            channel_first : Whether we want to make [2,H,W] or [H,W,2] meshgrid.
        '''
        h, w = self.image_size
        mesh_grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy'), axis=0 if channel_first else -1)
        if tensor_out:
            mesh_grid = torch.from_numpy(mesh_grid)
        return self.image2ray(mesh_grid, channel_first)

    def convert_from_cam(self, x: TensorLike, source_cam: CameraBase,
                         channel_first: bool=False, store_cvtFunction: bool=False) -> TensorLike:
        '''
            Convenience function to compute AND immediately remap an image to *THIS* camera
            x: Input image to be mapped
            source_cam: See get_converter_function. Source camera.
            channel_first : Whether the input image is [3,H,W] or [H,W,3]
            store_cvtFunction : Whether to store the resulting function. See get_converter_function for details.
        '''
        cvt_func = self.get_converter_function(source_cam, store_cvtFunction)
        return cvt_func(x, channel_first)

    def crop_convert(self, x: TensorLike, channel_first, TL_target, TL_source, H, W) -> TensorLike:
        '''
            Used to map between pinhole cameras of same extrinsic and same focal length. Faster than resampling.
            x : input image
            channel_first : Whether [3,H,W] or [H,W,3]
            TL_target : top_left corner in *THIS* image space
            TL_source : top_left corner on the source image space (of the 'source_cam')
            H,W : The size of the intersecting rectangle
        '''
        if channel_first:
            target_img_dims = [x.shape[0],*self.image_size]
        else:
            target_img_dims = [*self.image_size, x.shape[-1]]

        canvas = np.ones(target_img_dims, np.uint8)

        if channel_first:
            canvas[:, TL_target[0]:TL_target[0] + H, TL_target[1]:TL_target[1] + W] = x[:, TL_source[0]:TL_source[0] + H, TL_source[1]:TL_source[1] + W]
        else:
            canvas[TL_target[0]:TL_target[0] + H, TL_target[1]:TL_target[1] + W] = x[TL_source[0]:TL_source[0] + H, TL_source[1]:TL_source[1] + W]

        return canvas

    def _degree_to_pixel(self, vertical_deg=None, horizontal_deg=None):
        vertical_px = None if vertical_deg is None else (self.image_size[0] * (vertical_deg - self.vertical_fov_limits_deg[0]) /
                                                         (self.vertical_fov_limits_deg[1] - self.vertical_fov_limits_deg[0]))
        horizontal_px = None if horizontal_deg is None else (self.image_size[1] * (horizontal_deg - self.horizontal_fov_limits_deg[0]) /
                                                             (self.horizontal_fov_limits_deg[1] - self.horizontal_fov_limits_deg[0]))
        return vertical_px, horizontal_px

    def _pixel_to_degree(self, vertical_px=None, horizontal_px=None):
        vertical_deg = None if vertical_px is None else self.vertical_fov_limits_deg[0] + (
                self.vertical_fov_limits_deg[1] - self.vertical_fov_limits_deg[0]) * vertical_px / self.image_size[0]
        horizontal_deg = None if horizontal_px is None else self.horizontal_fov_limits_deg[0] + (
                self.horizontal_fov_limits_deg[1] - self.horizontal_fov_limits_deg[0]) * horizontal_px / self.image_size[1]
        return vertical_deg, horizontal_deg

if __name__ == '__main__':

    import cv2
    import time

    # region check conversion Pinhole -> Equirect
    intrinsic_mat = np.array([[1050, 0, 512],
                              [0, 1050, 84],
                              [0, 0, 1]])

    pinhole2equirect_runtimes = []
    equirect2pinhole_runtimes = []
    for ry in range(-30, 30, 1):
        # img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0054763.jpg')
        img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0006395.jpg')
        img_og = np.flipud(np.transpose(img_og, [1, 0, 2]))

        # channel last
        ch_first = False
        img = img_og

        # channel first
        # img = img_og.transpose([2, 0, 1])
        # ch_first = True

        # tensor conversion
        # img = torch.from_numpy(img.copy())

        source_rotation = Rotation.from_euler('yxz', [0, 0, 0], True)
        source_intrinsic = intrinsic_mat
        source_intrinsic[1, 2] = 390
        source_cam = CameraPinhole(source_intrinsic, img_og.shape[0:2], rotation=source_rotation, translation=[0, 0, 0])

        # target_rotation = Rotation.from_euler('yxz', [ry, 0, 0], degrees=True)
        target_rotation = Rotation.from_euler('yxz', [0, ry, 0], degrees=True)
        # target_rotation = Rotation.from_euler('yxz', [0, 0, ry], degrees=True)
        target_intrinsic = intrinsic_mat
        target_intrinsic[1, 2] = 55
        target_img_size = [256, 1024]
        start = time.time()
        target_cam = CameraEquirect(horizontal_fov_limits_deg=[-30,30], vertical_fov_limits_deg=[-15,15], image_size=target_img_size, rotation=target_rotation, translation=None)
        target_img = target_cam.convert_from_cam(img, source_cam, channel_first=ch_first, store_cvtFunction=True)
        end = time.time()
        pinhole2equirect_runtimes.append(end - start)
        start = time.time()
        resourced_img = source_cam.convert_from_cam(target_img, target_cam, ch_first, store_cvtFunction=True)
        end = time.time()
        equirect2pinhole_runtimes.append(end-start)

        if isTensor(target_img):
            target_img = target_img.numpy()

        if ch_first:
            target_img = target_img.transpose([1, 2, 0])

        cv2.imshow("Target crop", target_img)
        cv2.waitKey(10)

    pinhole2equirect_runtimes = np.asarray(pinhole2equirect_runtimes)
    equirect2pinhole_runtimes = np.asarray(equirect2pinhole_runtimes)
    print(f'Equirect <- Pinhole avg runtimes : {pinhole2equirect_runtimes.mean()}')
    print(f'Equirect -> Pinhole avg runtimes : {equirect2pinhole_runtimes.mean()}')
    # endregion

    # # region extensive testing
    # intrinsic_mat = np.array([[1050,    0, 512],
    #                           [   0, 1050,  84],
    #                           [   0,    0,   1]])
    #
    # # OG_img = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0054763.jpg')
    # OG_img = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0006395.jpg')
    # OG_img = np.flipud(np.transpose(OG_img, [1, 0, 2]))
    #
    # source_rotation = Rotation.from_euler('yxz', [0, 0, 0], True)
    # source_intrinsic = intrinsic_mat
    # source_intrinsic[1, 2] = 390
    #
    # horizontal_fov_test_done = False
    # vertical_fov_test_done = False
    # yaw_test_done = False
    # pitch_test_done = False
    # roll_test_done = False
    # channel_first_test_done = False
    # channel_last_test_done = False
    # tensor_type_test_done = False
    # array_type_test_done = False
    #
    # tensor_tests = [tensor_type_test_done, array_type_test_done]
    # while not all(tensor_tests):
    #     switch = tensor_tests.index(False)
    #     tensor_type = [True,False][switch]
    #     tensor_tests[switch] = True
    #     print(f"tensor_type : {tensor_type}, {switch}, {tensor_tests}")
    #     channel_tests = [channel_first_test_done, channel_last_test_done]
    #     while not all(channel_tests):
    #         switch = channel_tests.index(False)
    #         channel_first = [True, False][switch]
    #         print(f"channel_first: {channel_first}, {switch}, {channel_tests}")
    #         channel_tests[switch] = True
    #
    #         fov_tests = [horizontal_fov_test_done, vertical_fov_test_done]
    #         horizontal_fov_test_range = iter(np.arange(30, 0.5, -3.5))
    #         vertical_fov_test_range = iter(np.arange(20, 0.5, -3.5))
    #         while not all(fov_tests):
    #             switch = fov_tests.index(False)
    #
    #             if switch == 0:
    #                 vertical_fov = [-5,5]
    #                 try:
    #                     horizontal_test_fov = next(horizontal_fov_test_range)
    #                 except StopIteration:
    #                     fov_tests[switch] = True
    #                     continue
    #                 horizontal_fov = [-horizontal_test_fov, horizontal_test_fov]
    #
    #
    #             if switch == 1:
    #                 horizontal_fov = [-27,27]
    #                 try:
    #                     vertical_test_fov = next(vertical_fov_test_range)
    #                 except StopIteration:
    #                     fov_tests[switch] = True
    #                     continue
    #                 vertical_fov = [-vertical_test_fov, vertical_test_fov]
    #
    #             print(f"vfov : {vertical_fov}, hfov : {horizontal_fov}, {switch}, {fov_tests}")
    #
    #             ypr_tests = [yaw_test_done, pitch_test_done, roll_test_done]
    #             yaw_range = iter(np.arange(-10, 10, 0.5))
    #             pitch_range = iter(np.arange(-5, 5, 0.5))
    #             roll_range = iter(np.arange(-10, 10, 0.5))
    #             while not all(ypr_tests):
    #                 switch = ypr_tests.index(False)
    #                 ypr_gens = [yaw_range, pitch_range, roll_range]
    #                 ypr = [0,0,0]
    #                 try:
    #                     ypr[switch] = next(ypr_gens[switch])
    #                 except StopIteration:
    #                     ypr_tests[switch] = True
    #                     print(f'test done {switch}')
    #                     continue
    #
    #                 # channel last
    #                 if channel_first:
    #                     img = OG_img.transpose([2, 0, 1])
    #                     ch_first = True
    #                 else:
    #                     ch_first = False
    #                     img = OG_img
    #
    #                 if tensor_type:
    #                     img = torch.from_numpy(img.copy())
    #
    #                 print(ypr, vertical_fov, horizontal_fov, channel_first, tensor_type)
    #
    #                 source_cam = CameraPinhole(source_intrinsic, OG_img.shape[0:2], rotation=source_rotation, translation=[0, 0, 0])
    #                 target_cam = CameraEquirect(horizontal_fov,vertical_fov, image_size=[256,1024], rotation=Rotation.from_euler('yxz',ypr,True), translation=None)
    #                 target_img = target_cam.convert_from_cam(img, source_cam, channel_first=ch_first, store_cvtFunction=True)
    #
    #                 if isTensor(target_img):
    #                     target_img = target_img.numpy()
    #
    #                 if ch_first:
    #                     target_img = target_img.transpose([1, 2, 0])
    #
    #                 cv2.imshow("Target crop", target_img)
    #                 cv2.waitKey(1)
    #
    # # endregion
    print(target_cam.save_to_dict())
