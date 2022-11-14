#     from CameraBase import CameraBase, TensorLike, isTensor, isNumpy
# else:
from utils.sensor_models.CameraBase import CameraBase, TensorLike, isTensor, isNumpy
from typing import Union, Tuple, Callable
from functools import partial
from scipy.spatial.transform import Rotation

import torch.nn.functional
import cv2
import numpy as np
import torch
import json

class CameraPinhole(CameraBase):
    '''
        Undistorted pinhole camera.
        parameters:
            - intrinsic : 3x3 standard intrinsic matrix (aka ray2image)
                            [Fx, 0, Px]
                            [ 0,Fy, Py]
                            [ 0, 0,  1]
            - image_size : [Vertical Size, Horizontal Size]

            Extrinsics arguments:
            - rotation : rotation component in general scipy.spatial.transform.Rotation
                         It describes the orientation of the camera relative to the world/rig/body
            - translation: translation component
                         It desctibes the orientation of the camera relative to the world/rig/body

            Both extrinsic and its inverse gets computed from this.

            Note: in the camera we use Z-forward frame and the axis swap, if any, must be in the rotation part.
    '''

    model_name = 'pinhole'

    def __init__(self, intrinsic: np.ndarray, image_size: Union[list,tuple], rotation: Rotation = None, translation: Union[list, np.ndarray] = None):
        super().__init__(rotation, translation)
        self.model_name = CameraPinhole.model_name
        assert intrinsic.shape == (3,3)
        self.intrinsic = intrinsic.astype(np.float32)
        self.image_size = image_size

    # region MUST HAVES
    def image2ray(self, x: TensorLike, channel_first: bool = False) -> TensorLike:
        '''
            See CameraBase.image2ray.

            x : input coordinates to project (like the image meshgrid)
            channel_first : Whether the input is [2, ...] or [..., 2]. Output will be the same order.

            returns : array or tensor according to arguments. Same shape as 'x'.
        '''

        if not channel_first:
            x = torch.movedim(x, -1, 0) if isTensor(x) else np.moveaxis(x, -1, 0)

        x_shape = x.shape

        x = CameraPinhole.make_homogen(x).reshape(3,-1)
        rays = (CameraPinhole.invert_intrinsic(self.intrinsic) @ x).reshape(3, *x_shape[1:])

        if not channel_first:
            rays = torch.movedim(rays, 0, -1) if isTensor(x) else np.moveaxis(rays, 0, -1)

        return rays

    def ray2image(self, x: TensorLike, channel_first: bool = False) -> Tuple[TensorLike, TensorLike]:
        '''
            See CameraBase.ray2image

            x: The points to project onto the image plane.
            channel_first : Whether 'X' is [3, ...] or [..., 3]

            return:
                x_image : array or tensor of projected coordinates
                invalid_projection_mask : array. Where the projection failed or otherwise became invalid (eg.: Points from behind the focus point)
        '''
        if not channel_first:
            x = torch.movedim(x,-1,0) if isTensor(x) else np.moveaxis(x, -1, 0)
        x_shape = x.shape

        # Z check for points behind focal point.
        invalid_projection_mask = x[2:3] <= 0

        x_normed = x / x[2:3]
        intrinsic = torch.from_numpy(self.intrinsic) if isTensor(x) else self.intrinsic
        x_image = (intrinsic @ x_normed.reshape(3, -1)).reshape(x_shape)
        x_image = x_image[:2]

        if not channel_first:
            x_image = torch.movedim(x_image, 0, -1) if isTensor(x) else np.moveaxis(x_image, 0, -1)
            invalid_projection_mask = torch.movedim(invalid_projection_mask, 0, -1) if isTensor(x) else np.moveaxis(invalid_projection_mask, 0, -1)

        return x_image, invalid_projection_mask


    def get_converter_function(self, source_cam: CameraBase, store_cvtFunction: bool = False) -> Callable[[TensorLike, bool], TensorLike]:
        '''
            Computes a function that can map FROM the SOURCE camera to *THIS* (TARGET) camera.
            source_cam : Source camera.
            store_cvtFunction : Whether to cache the resulting function. If set to TRUE the next time we want to compute a mapping function
                                between two cameras that use the exact same parameters it will skip the computation and
                                return with the precomputed function. The cache is not part of the instance but the class!
                                Use it if you know you will probably remap between a few sets of cameras.
                                WARNING : There is a potential, that some special cases the function table can grow quite large.
        '''

        # if self.is_incomplete():
        #     self.apply_extrinsic_inheritance(source_cam)

        if not (self, source_cam) in CameraBase.cvtFunctionTable:
            # Insert : Simple heuristics to avoid resample if possible.
            # !WIP!
            if self == source_cam:
                cvtFunction = lambda x,y : x
            elif type(source_cam) == CameraPinhole and type(self) == CameraPinhole and \
                    source_cam.intrinsic[0, 0] == self.intrinsic[0, 0] and source_cam.intrinsic[1, 1] == self.intrinsic[1, 1] \
                    and np.all(self.RT_body_cam == source_cam.RT_body_cam):
                # can use crop
                Px = self.intrinsic[0, 2]
                Py = self.intrinsic[1, 2]
                Px_other = source_cam.intrinsic[0, 2]
                Py_other = source_cam.intrinsic[1, 2]

                TL = np.array([-Py, -Px], dtype=np.int32)
                H,W = self.image_size
                TL_O = np.array([-Py_other, -Px_other], dtype=np.int32)
                H_O, W_O = source_cam.image_size

                TL_intersection = np.maximum(TL, TL_O)
                H_intersection = min(TL[0]+H, TL_O[0]+H_O) - TL_intersection[0]
                W_intersection = min(TL[1]+W, TL_O[1]+W_O) - TL_intersection[1]

                if H_intersection <= 0 or W_intersection <= 0:
                    cvtFunction = lambda x, channel_first: np.zeros([x.shape[0],*self.image_size]) if channel_first else np.zeros([*self.image_size,x.shape[-1]])
                else:
                    TL_source = TL_intersection + np.asarray([Py_other, Px_other],np.int32)
                    TL_target = TL_intersection + np.asarray([Py, Px], np.int32)
                    cvtFunction = partial(self.crop_convert, TL_target=TL_target, TL_source=TL_source, H=H_intersection, W=W_intersection)

            else:
                grid_rays = self.grid2ray(channel_first=False)
                if not np.all(self.RT_body_cam == source_cam.RT_body_cam):
                    assert np.all(self.RT_body_cam[:3, 3] == source_cam.RT_body_cam[:3, 3]), \
                        "Camera conversion is invalid. The translation of the extrinsic between SOURCE and TARGET camera must match!"
                    extrinsic_transform = source_cam.RT_cam_body @ self.RT_body_cam
                    grid_rays = (extrinsic_transform @ self.make_homogen(grid_rays.transpose([2,0,1])).reshape([4,-1]))[:3].transpose([1,0]).reshape(grid_rays.shape)

                grid_mapping_other, validity_mask = source_cam.ray2image(grid_rays.astype(np.float32), channel_first=False)
                cvtFunction = partial(CameraPinhole.resample, mapping=grid_mapping_other, invalid_projection_mask=validity_mask)

            if store_cvtFunction:
                CameraBase.store_cvtFunction(self, source_cam, cvtFunction)

        else:
            cvtFunction = CameraBase.cvtFunctionTable[(self, source_cam)]

        return cvtFunction

    @classmethod
    def load_from_dict(cls, d:dict) -> 'CameraPinhole':
        '''Check CameraBase load_from_dict'''
        cam = super().load_from_dict(d)
        cam.intrinsic = np.asarray(cam.intrinsic, np.float32)

        return cam
    # endregion

    # region NICE TO HAVES
    def grid2ray(self, tensor_out: bool = False, channel_first: bool = False) -> TensorLike:
        '''
            Convenience function to project the image mesh grid.
            tensor_out : Whether we want the results in torch.Tensor or np.ndarray
            channel_first : Whether we want to make [2,H,W] or [H,W,2] meshgrid.
        '''
        h, w = self.image_size
        mesh_grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy'), axis=0 if channel_first else -1)
        if tensor_out:
            mesh_grid = torch.from_numpy(mesh_grid)
        return self.image2ray(mesh_grid, channel_first)
    # endregion

    # CONVERSION SHORTCUT FUNCTIONS
    def crop_convert(self, x: TensorLike, channel_first: bool, TL_target, TL_source, H, W) -> TensorLike:
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

    @staticmethod
    def make_homogen(x: TensorLike) -> TensorLike:
        '''
        Turns Nd tensor into homogen Nd. By adding the trailing ones.
        [3,H,W] containing [X,Y,Z] values -> [4,H,W] containing [X,Y,Z,1] values
        Now you can matrix multiply with 4x4 extrinsic matrix
        '''
        if isTensor(x):
            return torch.cat([x, torch.ones_like(x[0:1])], dim=0)
        elif isNumpy(x):
            return np.concatenate([x, np.ones_like(x[0:1])], axis=0)
        else:
            raise ValueError

    @staticmethod
    def invert_intrinsic(intrinsic: np.ndarray) -> TensorLike:
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        px = intrinsic[0, 2]
        py = intrinsic[1, 2]
        ret = np.array([[1 / fx, 0, -px / fx],
                        [0, 1 / fy, -py / fy],
                        [0, 0, 1]], np.float32)
        return ret

    @staticmethod
    def load_from_json_old(json_path: str, img: TensorLike, channel_first: bool = False):
        '''
        !!DEPRECATED!!
        Make camera out of calibration.json.
        Note : Since the calibration does not contain the height and width of the image in pixels we need to ask for
        the actual image associated with the calibration.json to complete the parameters
        '''
        if channel_first:
            c,h,w = img.shape
        else:
            h,w,c = img.shape

        with open(json_path, 'r') as filestream:
            content = json.load(filestream)
            intrinsic = content['ray_to_image']
            extrinsic = content['RT_cam_body']
            rotation = Rotation.from_matrix(extrinsic[:3, :3])
            translation = extrinsic[:3, 3]
            image_size = [h, w]
            return CameraPinhole(intrinsic, image_size, rotation, translation)

    # WIP - Not yet enabled
    # def load_from_json(fp) -> 'CameraPinhole':
    #     with open(fp, 'r') as stream:
    #         json_dict = json.load(stream)
    #     return CameraPinhole.load_from_dict(json_dict)

    # region OPTIONALS
    # Useful utility functions
    @staticmethod
    def functional_image2ray(x: TensorLike, intrinsic: np.ndarray, channel_first: bool) -> TensorLike:
        '''
        Functional stateless version of image2ray
        '''
        if channel_first:
            c,h,w = x.shape
        else:
            h,w,c = x.shape

        mesh_grid = np.stack(np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij'), axis=0)
        mesh_grid = CameraPinhole.make_homogen(mesh_grid).reshape([3, -1])
        rays = (CameraPinhole.invert_intrinsic(intrinsic) @ mesh_grid).reshape(3, h, w)

        if not channel_first:
            rays = np.rollaxis(rays, 0, 3)

        if isTensor(x):
            rays = torch.from_numpy(rays)

        return rays

    @staticmethod
    def functional_ray2image(x: TensorLike, intrinsic: np.ndarray, channel_first: bool) -> TensorLike:
        '''
        Functional stateless version ray2image
        '''
        if channel_first:
            c, h, w = x.shape
        else:
            h, w, c = x.shape
            x = x.permute(2, 0, 1) if isTensor(x) else np.transpose(x, [2, 0, 1])

        x_normed = x / x[2:3]
        intrinsic = torch.from_numpy(intrinsic) if isTensor(x) else intrinsic
        x_image = (intrinsic @ x_normed.reshape(3, -1)).reshape(3, h, w)
        x_image = x_image[:2]

        if not channel_first:
            x_image = x_image.permute(1, 2, 0) if isTensor(x) else np.transpose(x_image, [1, 2, 0])

        return x_image
    # endregion

if __name__ == '__main__':
    import time
    from scipy.spatial.transform import Rotation

    extrinsic_RT = np.eye(4)
    intrinsic_mat = np.array([[1050,   0, 512],
                              [   0,1050,  84],
                              [   0,   0,   1]])
    img_size = [1024, 269]

    r = Rotation.from_euler('yxz', [0,0,0], True)
    camera = CameraPinhole(intrinsic=intrinsic_mat, image_size=img_size, rotation=r, translation=[0, 0, 0])
    # # region check conversion
    # runtimes = []
    # for ry in range(-30,30,1):
    #     # img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0054763.jpg')
    #     img_og = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0006395.jpg')
    #     img_og = np.flipud(np.transpose(img_og, [1,0,2]))
    #
    #     # channel last
    #     # ch_first = False
    #     # img = img_og
    #
    #     # channel first
    #     img = img_og.transpose([2,0,1])
    #     ch_first = True
    #
    #     # tensor conversion
    #     # img = torch.from_numpy(img.copy())
    #
    #     source_rotation = Rotation.from_euler('yxz',[0,0,0], True)
    #     source_intrinsic = intrinsic_mat
    #     source_intrinsic[1, 2] = 390
    #     source_cam = CameraPinhole(source_intrinsic, img_og.shape[0:2], rotation=source_rotation, translation=[0, 0, 0])
    #
    #     target_rotation = Rotation.from_euler('yxz', [0, ry, 0], degrees=True)
    #     target_intrinsic = intrinsic_mat
    #     target_intrinsic[1,2] = 55
    #     target_img_size = [256,1024]
    #     start = time.time()
    #     target_cam = CameraPinhole(target_intrinsic, target_img_size, rotation=target_rotation, translation=None)
    #     target_img = target_cam.convert_from_cam(img, source_cam, channel_first=ch_first, store_cvtFunction=True)
    #     end = time.time()
    #     runtimes.append(end - start)
    #
    #     if isTensor(target_img):
    #         target_img = target_img.numpy()
    #
    #     if ch_first:
    #         target_img = target_img.transpose([1,2,0])
    #
    #     cv2.imshow("Target crop", target_img)
    #     cv2.waitKey(10)
    #
    # runtimes=np.asarray(runtimes)
    # print(f'avg runtimes : {runtimes.mean()}')
    # # endregion

    # region check cached transforms
    # runtimes = []
    # img = cv2.imread('/home/ad.adasworks.com/domonkos.kovacs/Work/bev/src/imgproc/sensor_models/0054763.jpg')
    # img = np.flipud(np.transpose(img, [1, 0, 2]))
    # source_extrinsic = extrinsic_RT
    # source_intrinsic = intrinsic_mat
    # source_intrinsic[1, 2] = 390
    # source_cam = CameraPinhole(source_extrinsic, source_intrinsic, img.shape[0:2])
    #
    # target_intrinsic = intrinsic_mat
    # target_intrinsic[1, 2] = 55
    # target_img_size = [148, 1024]
    # target_cam = CameraPinhole(source_extrinsic, target_intrinsic, target_img_size)
    #
    # for i in range(-500, 500, 1):
    #     start = time.time()
    #     target_img = target_cam.convert_from_cam_model(img, source_cam, False, False)
    #     end = time.time()
    #     runtimes.append(end - start)
    #     cv2.imshow("Target crop", target_img)
    #     cv2.waitKey(1)
    #
    # runtimes = np.asarray(runtimes)
    # print(f'avg runtimes : {runtimes.mean()}')
    # endregion


