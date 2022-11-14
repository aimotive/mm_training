import json

import numpy as np
import torch
import torch.nn.functional
import cv2

from typing import Union, Callable, Tuple
from scipy.spatial.transform import Rotation

# region Helper typedefs
isTensor = lambda x: isinstance(x,torch.Tensor)
isNumpy = lambda x: isinstance(x,np.ndarray)
TensorLike = Union[np.ndarray, torch.Tensor]
# endregion

class CameraBase:

    '''
        Archetype Class. Any camera model can be derived from this for ease of use and better maintenance.
        Every camera model is expected to handle two simple operations.

        image2view : Transforms coordinates from image space into 3D camera space (aka rays captured by the image).
        view2image : Project 3D points (aka rays) from camera space onto the image space.

        Extrinsics arguments:
            - rotation : rotation component in general scipy.spatial.transform.Rotation
                         It describes the orientation of the camera relative to the world/rig/body
            - translation: translation component
                         It desctibes the orientation of the camera relative to the world/rig/body

            Both extrinsic and its inverse gets computed from this.

        Note: in the camera we use Z-forward frame and the axis swap, if any, must be in the rotation part.

        The class (not the instances) owns a static library. It can cache any requested
    '''

    def __init__(self, rotation: Rotation = None, translation: Union[list,np.ndarray] = None):
        # extrinsic precompute
        self.incomplete_rotation = rotation is None
        self.incomplete_translation = translation is None
        self.incomplete = rotation is None or translation is None

        self._RT_body_cam = np.eye(4, dtype=np.float32)
        if rotation is not None:
            self._RT_body_cam[:3, :3] = rotation.as_matrix()
        if translation is not None:
            self._RT_body_cam[:3,  3] = np.asarray(translation)

        self._RT_cam_body = np.eye(4, dtype=np.float32)
        self._RT_cam_body[:3, :3] = self._RT_body_cam[:3, :3].T
        self._RT_cam_body[:3,  3] = -(self._RT_body_cam[:3, :3].T @ self._RT_body_cam[:3, 3])

    @property
    def RT_body_cam(self):
        return self._RT_body_cam

    @property
    def RT_cam_body(self):
        return self._RT_cam_body

    def image2ray(self, x: TensorLike, channel_first: bool=False) -> TensorLike:
        raise NotImplementedError

    def ray2image(self, x: TensorLike, channel_first: bool=False) -> Tuple[TensorLike, TensorLike]:
        raise NotImplementedError

    def get_converter_function(self, source_cam: 'CameraBase', store_cvtFunction=False) -> Callable[[TensorLike, bool], TensorLike]:
        raise NotImplementedError

    def convert_from_cam(self, x: TensorLike, source_cam: 'CameraBase',
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

    def apply_extrinsic_inheritance(self, source_cam: 'CameraBase') -> None:
        if self.incomplete_rotation and not source_cam.incomplete_rotation:
            self._RT_body_cam[:3, :3] = source_cam._RT_body_cam[:3, :3]
            self._RT_cam_body[:3, :3] = source_cam._RT_cam_body[:3, :3]
            self.incomplete_rotation = False
        if self.incomplete_translation and not source_cam.incomplete_translation:
            self._RT_body_cam[:3, 3] = source_cam._RT_body_cam[:3, 3]
            self._RT_cam_body[:3, 3] = source_cam._RT_cam_body[:3, 3]
            self.incomplete_translation = False
        self.incomplete = self.incomplete_rotation or self.incomplete_translation

        return

    @staticmethod
    def resample(y: TensorLike, channel_first: bool, mapping: TensorLike, invalid_projection_mask: TensorLike):
        '''
            Uses cv2.resample or torch.grid_sample depending on the input type.
            y : SOURCE image to be remapped
            channel_first : whether [3,H,W] or [H,W,3]
            mapping: [H,W,2] shaped float tensor or array
            invalid_projection_mask : some of the mapping associations may yield invalid results. (eg.: point behind the focal point in case of Pinhole)
        '''
        if isTensor(y):
            if channel_first:
                c, h, w = y.shape
            else:
                h, w, c = y.shape
                y = y.permute(2, 0, 1)
            # Tensor route - better convert everything to channel first
            invalid_projection_mask = torch.from_numpy(invalid_projection_mask[...,0])
            normalizer = torch.tensor([(2.0/w), (2.0/h)]).reshape(1, 1, 2)
            mapping_t = torch.from_numpy(mapping)
            mapping_t = mapping_t * normalizer - 1
            resampled_image = torch.nn.functional.grid_sample(y.unsqueeze(0).float(), mapping_t.unsqueeze(0)).byte()[0]
            resampled_image[:, invalid_projection_mask] = 0
            if not channel_first:
                resampled_image = resampled_image.permute(1, 2, 0)
        else:
            # Hint: we can use cv2.convertMaps for somewhat increased speed and smaller memory footprint
            if channel_first:
                y = np.transpose(y, [1, 2, 0])
            resampled_image = cv2.remap(y, mapping[...,0], mapping[...,1], cv2.INTER_LINEAR)
            resampled_image[np.repeat(invalid_projection_mask, y.shape[-1], -1)] = 0
            if channel_first:
                resampled_image = np.transpose(resampled_image, [2, 0, 1])

        return resampled_image



    # region SAVE and LOAD functions

    def save_to_dict(self) -> dict:
        '''
            Returns dict representation of the instance
        '''
        return self.__dict__

    def save_to_string(self) -> str:
        '''
            Returns JSON string representation of the instance
        '''
        d = self.save_to_dict()
        string = json.dumps(d, cls=CameraBase.NumpyEncoder)
        return string

    def save_to_json(self, fp) -> None:
        '''
            Saves the instance as a JSON file
            fp: path to output file
        '''
        s = self.save_to_string()
        with open(fp, 'w') as stream:
            stream.write(s)

    @classmethod
    def load_from_dict(cls, d: dict):
        '''
            Instantiates the approriate derived class and uses the dict to initialize its variables
            IMPORTANT!
                Every derived class must handle the LIST->nd.array casting in its derived load_from_dict function.
                That is. JSON cannot contain nd.array. During serialization these are converted into nested lists.
                These are read back again az nested lists. For instance in the Pinhole the intrinsic is an nd.array.
                In its load_from_dict it calls *THIS* function (as it's parent method) then casts it's own .intrinsic variable
                to np.ndarray.
        '''
        cl = cls.__new__(cls)
        for k,v in d.items():
            cl.__setattr__(k, v)

        cl._RT_body_cam = np.asarray(cl._RT_body_cam, np.float32)
        cl._RT_cam_body = np.asarray(cl._RT_cam_body, np.float32)

        return cl

    # WIP - Not yet enabled
    # @classmethod
    # def load_from_json(fp):
    #     raise NotImplementedError

    # endregion

    # region Under the Hood features
    '''
        Caching as of 2022.06.16
        Currently the caching is based on the "str" representation of the contents of the objects involved in the conversion.
        The resulting conversion function is invariant to the TRANSLATION and ROTATION of the source cam and only varies for the DELTA between target and source.
        Since the TRANSLATION is currently assumed or explicitly held equal we can ommit it from the check. (No sensors support it currently)
        ROTATION is currently not ommited (or replaced by the DELTA ROTATION) only because there are only a few rotation variants in use.   
    '''

    cvtFunctionTable = {}

    @staticmethod
    def store_cvtFunction(target, source, cvtFunction):
        CameraBase.cvtFunctionTable[target,source] = cvtFunction

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, torch.Tensor):
                return o.numpy().tolist()
            else:
                return json.JSONEncoder.default(self, o)

    def __repr__(self):
        return ';'.join([str(self.__getattribute__(attr_name)) if attr_name not in ['_RT_body_cam', '_RT_cam_body']
                         else str(self.__getattribute__(attr_name)[:3,:3]) for attr_name in self.__dict__])

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__hash__() == hash(other)
    # endregion
