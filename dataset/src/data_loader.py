import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dataset.src.annotation import Annotation
from dataset.src.camera_params import CameraParams
from dataset.src.loaders.camera_loader import load_camera_data, CameraData, CameraDataItem
from dataset.src.loaders.lidar_loader import load_lidar_data, LidarData
from dataset.src.loaders.radar_loader import load_radar_data, RadarData
from utils.sensor_models import CameraPinholeDistorted, CameraMei
from utils.sensor_models.CameraPinhole import CameraPinhole


class DataItem:
    """
    Data structure for storing data for a given frame id.

    Attributes:
        annotations: Annotation instance
        lidar_data: LidarData instance
        radar_data: RadarData instance
        camera_data: CameraData instance
    """
    def __init__(self, annotations: Annotation, lidar_data: LidarData, radar_data: RadarData, camera_data: CameraData):
        self.annotations = annotations
        self.lidar_data = lidar_data
        self.radar_data = radar_data
        self.camera_data = camera_data


# R_Z_forward_to_body = Rotation.from_matrix(np.array([[0, 0, 1],
#                                                      [-1, 0, 0],
#                                                      [0, -1, 0]], dtype=np.float32))

R_Z_forward_to_body = Rotation.from_matrix(np.array([[0, 0, 1],
                                                     [-1, 0, 0],
                                                     [0, -1, 0]], dtype=np.float32))


class DataLoader:
    """
    Loads sensor data for a given frame id.

    Attributes:
        data_paths: a list of keyframe paths
    """
    CATEGORY_MAPPING = {'CAR': 0, 'Size_vehicle_m': 0,
                        'TRUCK': 1, 'BUS': 1, 'TRUCK/BUS': 1, 'TRAIN': 1, 'Size_vehicle_xl': 1, 'VAN': 1,
                        'PICKUP': 1,
                        'MOTORCYCLE': 2, 'RIDER': 2, 'BICYCLE': 2, 'BIKE': 2, 'Two_wheel_without_rider': 2,
                        'Rider': 2,
                        'OTHER_RIDEABLE': 2, 'OTHER-RIDEABLE': 2,
                        'PEDESTRIAN': 3, 'BABY_CARRIAGE': 3, 'SHOPPING-CART': 4, 'OTHER-OBJECT': 4, 'TRAILER': 1
                        }

    def __init__(self, data_paths, split, pc_range, use_cam=True, use_lidar=True, use_radar=True, look_back=0, look_forward=0):
        """
        Args:
            data_paths: a list of keyframe paths
        """
        self.pc_range = pc_range
        self.use_radar = use_radar
        self.look_forward = look_forward
        self.look_back = look_back
        self.split = split
        self.use_lidar = use_lidar
        self.use_cam = use_cam
        self.data_paths = data_paths
        self.MAX_LIDAR_POINTS = (look_back + look_forward + 1) * 100_000

    def __getitem__(self, path: str) -> DataItem:
        """
        Returns sensor data for a given keyframe.

        Args:
            path: path of the keyframe's annotation file

        Returns:
            a DataItem with annotations and sensor data
        """

        data_folder = self.get_directory(path)
        frame_id = self.get_frame_id(path)
        annotations = Annotation(path)
        lidar_data = load_lidar_data(data_folder, frame_id, self.look_back, self.look_forward)
        radar_data = load_radar_data(data_folder, frame_id) if self.use_radar else None
        camera_data = load_camera_data(data_folder, frame_id, self.use_cam)
        lidar_radar_data = self.concat_lidar_radar(lidar_data, radar_data, camera_data.timestamp) if self.use_radar else lidar_data.top_lidar.point_cloud

        lidar_radar_data = self.filter_lidar_data(lidar_radar_data)

        reference_intrinsic = camera_data.front_camera.camera_params.intrinsic
        zero_rot = self.use_cam

        if self.use_cam:
            camera_data.items = self.virtualize_cameras(camera_data, reference_intrinsic, zero_rot)

        lidar_timestamp_min = lidar_radar_data[:, -1].min()
        lidar_timestamp_max = lidar_radar_data[:, -1].max()
        lidar_radar_data[:, -1] = (lidar_radar_data[:, -1] - lidar_timestamp_min) / (lidar_timestamp_max - lidar_timestamp_min)
        camera_data.timestamp = (camera_data.timestamp - lidar_timestamp_min) / (lidar_timestamp_max - lidar_timestamp_min)

        lidar_radar_data = self.process_lidar(lidar_radar_data)
        lidar_data.top_lidar.point_cloud = lidar_radar_data

        extrinsics = [camera.camera_params.extrinsic for camera in camera_data.items]
        if self.use_cam and not self.use_lidar:
            annotations = self.filter_annotations_by_fov(annotations,
                                                         extrinsics
                                                         )


        # Making objects into one array
        objects = [self.object_to_array(obj) for obj in annotations.objects]
        objects = list(filter(lambda x: x[-1] in DataLoader.CATEGORY_MAPPING, objects))

        for obj in objects:
            obj[-1] = DataLoader.CATEGORY_MAPPING[obj[-1]]

        annotations.objects = torch.Tensor(objects)

        if self.use_lidar:
            annotations = self.filter_annotations_by_num_points(annotations, lidar_data)

        return DataItem(annotations, lidar_data, radar_data, camera_data)

    def filter_annotations_by_num_points(self, annotations: Annotation, lidar_data: LidarData) -> Annotation:
        pc = torch.Tensor(lidar_data.top_lidar.point_cloud)
        new_objects = list()

        for obj in annotations.objects:
            in_x = torch.logical_and(obj[0] - obj[3] / 2 <= pc[:, 0],
                              pc[:, 0] <= obj[0] + obj[3] / 2)
            in_y = torch.logical_and(obj[1] - obj[4] / 2 <= pc[:, 1],
                              pc[:, 1] <= obj[1] + obj[4] / 2)
            in_z = torch.logical_and(obj[2] - obj[5] / 2 <= pc[:, 2],
                                     pc[:, 2] <= obj[2] + obj[5] / 2)
            num_points = torch.logical_and(torch.logical_and(in_x, in_y), in_z).sum()

            if num_points > 5:
                new_objects.append(obj)

        if len(new_objects) > 0:
            annotations.objects = torch.stack(new_objects)
        else:
            annotations.objects = torch.zeros((0, 10))

        return annotations

    def virtualize_cameras(self, camera_data, reference_intrinsic, zero_rot):
        new_cam_list = list()
        for camera in camera_data.items:
            if camera.image is None:
                continue

            is_pinhole = 'front' in camera.name or 'back' in camera.name
            if is_pinhole:
                new_img, new_intrinsic, new_extrinsic = self.create_virtual_image(camera.image,
                                                                                  camera.camera_params,
                                                                                  reference_intrinsic,
                                                                                  zero_rot)
                params = CameraParams(new_intrinsic, camera.camera_params.extrinsic, camera.camera_params.dist_coeffs,
                                      camera.camera_params.camera_model, camera.camera_params.xi)
                new_camera = CameraDataItem(camera.name, new_img, params)
                new_cam_list.append(new_camera)
            else:
                yaw = self.get_yaw_from_params(camera.camera_params)

                img_left, img_left_intrinsic, img_left_extrinsic = self.create_virtual_image(camera.image,
                                                                                             camera.camera_params,
                                                                                             reference_intrinsic,
                                                                                             zero_rot,
                                                                                             yaw - 30
                                                                                             )

                img_right, img_right_intrinsic, img_right_extrinsic = self.create_virtual_image(camera.image,
                                                                                                camera.camera_params,
                                                                                                reference_intrinsic,
                                                                                                zero_rot,
                                                                                                yaw + 30)
                params_left = CameraParams(img_left_intrinsic, img_left_extrinsic, camera.camera_params.dist_coeffs,
                                           'opencv_pinhole', None)
                cam_left = CameraDataItem(camera.name, img_left, params_left)
                params_right = CameraParams(img_right_intrinsic, img_right_extrinsic, camera.camera_params.dist_coeffs,
                                            'opencv_pinhole', None)
                cam_right = CameraDataItem(camera.name, img_right, params_right)
                new_cam_list.append(cam_left)
                new_cam_list.append(cam_right)
        return new_cam_list

    def filter_lidar_by_timestamp(self, camera_data, lidar_data, delta=0.005):
        lidar_timestamps = lidar_data.top_lidar.point_cloud[:, 4] * 1e-9
        camera_timestamp = camera_data.timestamp * 1e-9
        pc = lidar_data.top_lidar.point_cloud
        pc_mask = np.logical_and(lidar_timestamps > camera_timestamp - delta, lidar_timestamps < camera_timestamp + delta)
        pc = pc[pc_mask]
        return pc

    def get_yaw_from_params(self, params: CameraParams) -> float:
        ext = np.linalg.inv(params.extrinsic)
        rot = Rotation.from_matrix(ext[:3, :3])
        rot_euler = (rot * R_Z_forward_to_body.inv()).as_euler('XYZ', degrees=True)
        return rot_euler[2]

    def create_virtual_image(self, img: np.ndarray,
                             params: CameraParams,
                             new_intrinsic: np.ndarray,
                             zero_roll_pitch: bool = False,
                             new_yaw: float = None
                             ):

        ext = np.linalg.inv(params.extrinsic)

        rot = Rotation.from_matrix(ext[:3, :3])
        translation = ext[:3, 3]

        if params.xi is None:
            source_cam = CameraPinholeDistorted(params.intrinsic[:, :3],
                                                params.dist_coeffs, [img.shape[0], img.shape[0]], rot, translation)
        else:
            source_cam = CameraMei(params.intrinsic[:, :3], params.xi, params.dist_coeffs, [img.shape[0], img.shape[0]],
                                   rot, translation)

        if zero_roll_pitch:
            rot_euler = (rot * R_Z_forward_to_body.inv()).as_euler('XYZ', degrees=True)
            rot_euler[[0, 1]] = 0
            rot_euler[2] = rot_euler[2] if new_yaw is None else new_yaw
            rot = Rotation.from_euler('XYZ', rot_euler, degrees=True) * R_Z_forward_to_body

        target_cam = CameraPinhole(new_intrinsic[:, :3], [704, 1280], rot, translation)

        out_img = target_cam.convert_from_cam(img, source_cam, store_cvtFunction=True)

        new_intrinsic = np.eye(4)
        new_intrinsic[:3, :3] = target_cam.intrinsic

        new_extrinsic = target_cam.RT_cam_body
        return out_img, new_intrinsic, new_extrinsic

    def object_to_array(self, object):
        x = object['BoundingBox3D Origin X']
        y = object['BoundingBox3D Origin Y']
        z = object['BoundingBox3D Origin Z']

        dx = object['BoundingBox3D Extent X']
        dy = object['BoundingBox3D Extent Y']
        dz = object['BoundingBox3D Extent Z']
        ori = Rotation.from_quat((object['BoundingBox3D Orientation Quat X'],
                                  object['BoundingBox3D Orientation Quat Y'],
                                  object['BoundingBox3D Orientation Quat Z'],
                                  object['BoundingBox3D Orientation Quat W'])).as_euler('xyz', degrees=False)[
            2]  # ???

        category_name = object['ObjectType']

        vx = object['Relative Velocity X']
        vy = object['Relative Velocity Y']
        return [x, y, z, dx, dy, dz, ori, vx, vy, category_name]

    def filter_annotations_by_fov(self, annotations: Annotation, extrinsics):
        def contains(x, y, fov=60):
            coef = np.tan((fov / 2.) * np.pi / 180)
            in_cone = (-coef * x < y) & (y < coef * x)
            return in_cone and x > 0.5

        new_objects = list()

        for obj in annotations.objects:
            obj_coords = [obj['BoundingBox3D Origin X'], obj['BoundingBox3D Origin Y'], obj['BoundingBox3D Origin Z'], 1]
            for extrinsic in extrinsics:
                obj_in_camera_coord = extrinsic @ obj_coords
                in_fov = contains(obj_in_camera_coord[2], obj_in_camera_coord[0])
                if in_fov:
                    new_objects.append(obj)

        annotations.objects = new_objects

        return annotations

    def get_directory(self, path: str) -> str:
        """
        Returns with the sequence directory of a given keyframe.

        Args:
            path: path of the keyframe's annotation file

        Returns:
            directory_path: path to the directory of the sequence which contains the given keyframe
        """
        directory_path = os.path.normpath(path).split(os.path.sep)[:-4]
        directory_path = os.path.sep.join(directory_path)

        return directory_path

    def get_frame_id(self, path: str) -> str:
        """
        Parses the frame id form a given path.

        Args:
            path: a path to a frame.

        Returns:
            frame_id: the parsed frame id
        """
        frame_id = os.path.normpath(path).split(os.path.sep)[-1]
        frame_id = os.path.splitext(frame_id)[0]
        frame_id = frame_id.split('_')[1]

        return frame_id

    def process_lidar(self, point_cloud: np.ndarray) -> np.ndarray:
        n_points, n_features = point_cloud.shape
        point_cloud[:, -2] /= 255.
        if n_points > self.MAX_LIDAR_POINTS:
            np.random.shuffle(point_cloud)
            new_point_cloud = point_cloud[:self.MAX_LIDAR_POINTS]
        else:
            new_point_cloud = point_cloud

        return new_point_cloud

    def concat_lidar_radar(self, lidar_data: LidarData, radar_data: RadarData, camera_timestamp: int) -> np.ndarray:
        lidar = lidar_data.top_lidar.point_cloud
        lidar = np.hstack([lidar[:, 0:3], np.zeros((lidar.shape[0], 1)), np.zeros((lidar.shape[0], 2)), np.expand_dims(lidar[:, 3], -1), np.expand_dims(lidar[:, 4], -1)])
        radar = np.vstack([radar_data.back_radar.point_cloud, radar_data.front_radar.point_cloud])
        radar = np.hstack([radar[:, 0:3], np.ones((radar.shape[0], 1)), radar[:, 3:], np.zeros((radar.shape[0], 1)), camera_timestamp * np.ones((radar.shape[0], 1))])
        radar_lidar = np.vstack([radar, lidar]).astype(np.float32)
        return radar_lidar

    def filter_lidar_data(self, pc: np.ndarray):
        in_x = np.logical_and(pc[:, 0] > self.pc_range[0], pc[:, 0] < self.pc_range[3])
        in_y = np.logical_and(pc[:, 1] > self.pc_range[1], pc[:, 1] < self.pc_range[4])
        in_z = np.logical_and(pc[:, 2] > self.pc_range[2], pc[:, 2] < self.pc_range[5])
        in_range = np.logical_and(in_x, in_y, in_z)
        return pc[in_range]



