from typing import Tuple, List, Dict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.annotation import Annotation
from src.camera_params import CameraParams
from src.data_loader import DataItem
from src.loaders.camera_loader import CameraData
from src.loaders.lidar_loader import LidarData
from src.loaders.radar_loader import RadarData


class Renderer:
    """
    A class for rendering sensor data and corresponding annotations for a given DataItem.
    """
    Color = Tuple[int, int, int]

    def __init__(self, save_images: bool = True, out_dir: Path = None):
        self.topdown_image_width = 2000
        self.topdown_image_height = 400
        self.forward_filter = 200
        self.side_filter = 40
        self.image_resolution = 20      # 1 pixel in lidar visualization image covers 20 centimeters in world space.
        self.image_resize_ratio = 60
        self.back_image_resize_ratio = 35
        self.topdown_image_resize_ratio = 100
        self.save_images = save_images
        self.render_idx = 0
        #self.out_dir = out_dir
        self.out_dir = Path('./output/pointpillar')
        if self.save_images:
            self.out_dir.mkdir(exist_ok=True, parents=True)


    def render(self, data: DataItem):
        """
        Renders sensor data for a given keyframe.

        Args:
            data: DataItem storing the multimodal sensor data and annotations
        """
        if data.camera_data:
            self.render_camera(data.camera_data, data.annotations)
        if data.radar_data:
            self.render_radar(data.radar_data, data.annotations)
        if data.lidar_data:
            self.render_lidar(data.lidar_data, data.annotations)

        self.render_idx += 1

        cv2.waitKey(1)

    def render_camera(self, camera_data: CameraData, annotation: Annotation):
        """
        Renders camera data for a given keyframe.

        Args:
            camera_data: CameraData storing sensor data for each camera
            annotation: Annotation instance, stores annotated dynamic objects
        """
        for camera in camera_data:
            img = camera.image
            camera_params = camera.camera_params

            img = self.plot_image_annotations(img, annotation, camera_params, camera.name)
            img = self.resize_image(camera.name, img)

            if self.save_images:
                save_path = str(self.out_dir / f'{camera.name}_{str(self.render_idx).zfill(7)}.jpg')
                cv2.imwrite(save_path, img)

            cv2.imshow(camera.name, img)

    def plot_image_annotations(self, img: np.array, annotation: Annotation, camera_params: CameraParams,
                               sensor_name: str) -> np.array:
        """
        Visualizes annotations on images corresponding to a given camera.

        Args:
            img: image to plot annotations
            annotation: Annotation instance, stores annotated dynamic objects
            camera_params: stores data required for projections.
            sensor_name: camera name

        Returns:
             img: image with visualized annotations
        """
        objs_in_fov = [obj for obj in annotation.objects if self.is_in_fov(obj, sensor_name)]
        img = self.project_annotations_to_image(img, objs_in_fov, camera_params)

        return img

    def resize_image(self, sensor: str, img: np.array) -> np.array:
        """
        Resizes visualization image based on sensor type.

        Args:
            sensor: sensor name
            img: image to be resized

        Returns:
            img: resized image
        """
        if sensor in ['radar', 'lidar']:
            scale_percent = self.topdown_image_resize_ratio
        else:
            scale_percent = self.image_resize_ratio if sensor != 'back_cam' else self.back_image_resize_ratio
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        return img

    def render_radar(self, radar_data: RadarData, annotation: Annotation):
        """
        Renders radar data for a given keyframe.

        Args:
            radar_data: RadarData storing sensor data for each radar
            annotation: Annotation instance, stores annotated dynamic objects
        """
        radar_pts = []
        img = np.zeros([self.topdown_image_height, self.topdown_image_width, 3])

        # Collect targets from all radars.
        for radar in radar_data:
            pcd = radar.point_cloud
            radar_pts.append(self.encode_pcd_to_image_grid(pcd))

        # Create point cloud for the whole perception area.
        radar_pts = (np.concatenate([radar_pts[0][0], radar_pts[1][0]]),
                     np.concatenate([radar_pts[0][1], radar_pts[1][1]]))

        img[radar_pts[0], radar_pts[1]] = (255, 255, 255)

        # Apply dilation for making radar targets in the image more visible.
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
        img = self.plot_topdown_annotation(img, annotation)
        # Flip is needed because of body coordinate system (x -> forward, y -> left, z -> top).
        img = np.flipud(img)
        img = self.resize_image('radar', img)
        cv2.imshow('Radar', img)

    def render_lidar(self, lidar_data: LidarData, annotation: Annotation):
        """
        Renders lidar data for a given keyframe.

        Args:
            lidar_data: LidarData storing sensor data for each lidar (currently one lidar is used)
            annotation: Annotation instance, stores annotated dynamic objects
        """
        sensor = lidar_data.top_lidar
        img = np.zeros([self.topdown_image_height, self.topdown_image_width, 3])

        pcd = sensor.point_cloud
        lidar_pts = self.encode_pcd_to_image_grid(pcd)
        img[lidar_pts[0], lidar_pts[1]] = (255, 255, 255)
        img = self.plot_topdown_annotation(img, annotation)
        img = self.resize_image('lidar', img)
        # Flip is needed because of body coordinate system (x -> forward, y -> left, z -> top).
        img = np.flipud(img)
        cv2.imshow(sensor.name, img)

    def encode_pcd_to_image_grid(self, pcd: np.array) -> Tuple[List[int], List[int]]:
        """
        Projects a point cloud from body coordinates into image coordinates.
        Body coordinate system: x -> forward, y -> left, z -> top
        Image coordinate system: OpenCV image coordinate system (https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/)

        Args:
            pcd: point cloud in body coordinates

        Returns:
            data_pts: tuple(list of y coordinates in image space, list of x coordinates in image space)
        """
        pcd = pcd[(pcd[:, 0] < self.forward_filter) & (pcd[:, 0] > -self.forward_filter)]
        pcd = pcd[(pcd[:, 1] < self.side_filter) & (pcd[:, 1] > -self.side_filter)]
        data_pts = ((pcd[:, 1] * (100 / self.image_resolution) + self.topdown_image_height / 2).astype(np.int32),
                    (pcd[:, 0] * (100 / self.image_resolution) + self.topdown_image_width / 2).astype(np.int32))

        return data_pts

    def plot_topdown_annotation(self, img: np.array, annotation: Annotation, color: Color = (255, 255, 255)):
        """
        Visualizes annotations in BEV (topdown) space.

        Args:
            img: topdown image with visualized sensor data
            annotation: Annotation instance, stores annotated dynamic objects
            color: color of visualizations

        Returns:
            img: topdown images with visualized sensor data and annotations
        """
        for obj in annotation.objects:
            x, y = obj['BoundingBox3D Origin X'], obj['BoundingBox3D Origin Y']
            x, y = self.encode_to_image_resolution(x), self.encode_to_image_resolution(y, forward_direction=False)
            width, length = obj['BoundingBox3D Extent Y'], obj['BoundingBox3D Extent X']
            # x, y are stored in meters, pixels in image represent centimeters, therefore we need to multiply x,y by 100.
            width, length = width * (100 / self.image_resolution), length * (100 / self.image_resolution)
            qw, qx = obj['BoundingBox3D Orientation Quat W'], obj['BoundingBox3D Orientation Quat X']
            qy, qz = obj['BoundingBox3D Orientation Quat Y'], obj['BoundingBox3D Orientation Quat Z']
            eps = 1e-3
            if np.linalg.norm(np.array((qw, qx, qy, qz))) < eps:
                qx, qy, qz, qw = (0, 0, 0, 1)

            corners = np.array([[-1, -1],
                                [-1, 1],
                                [1, 1],
                                [1, -1]
                                ]) * np.array([length, width]) / 2
            corners = np.array([x, y]) + (Rotation.from_quat((qx, qy, qz, qw)).as_matrix()[:2, :2] @ corners.T).T
            corners = corners.astype(np.int32)

            img = cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, color, thickness=2)
            img = cv2.line(img, (corners[2, 0], corners[2, 1]), (corners[3, 0], corners[3, 1]),
                           color=(255, 0, 0), thickness=2)

        return img

    def encode_to_image_resolution(self, value: float, forward_direction: bool = True) -> float:
        """
        Projects a point from body coordinate system into image coordinate system.
        Body coordinate system: x -> forward, y -> left, z -> top
        Image coordinate system: OpenCV image coordinate system.

        Args:
            value: 3D point in body coordinate system
            forward_direction: whether the coordinate is aligned with the forward axis

        Returns:
             value: 2D point in camera coordinate system
        """
        if forward_direction:
            value = value * (100 / self.image_resolution) + self.topdown_image_width / 2
        else:
            value = value * (100 / self.image_resolution) + self.topdown_image_height / 2

        return value

    def project_annotations_to_image(self, img: np.array, captured_objects: dict,
                                     camera_params: CameraParams) -> np.array:
        """
        Projects annotations to the camera image.

        Args:
            img: camera image where annotations are visualized
            captured_objects: dict of annotated objects, see details in Annotation.py
            camera_params: camera parameters needed for projection

        Returns:
            img: image with visualized annotations

        """
        intrinsic = camera_params.intrinsic
        extrinsic = camera_params.extrinsic
        fisheye = True if camera_params.camera_model == 'mei' else False

        corners = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5]
        ])

        for bbox in captured_objects:
            center = np.array([bbox[f'BoundingBox3D Origin {ax}'] for ax in ['X', 'Y', 'Z']])
            dims = np.array([bbox[f'BoundingBox3D Extent {ax}'] for ax in ['X', 'Y', 'Z']])
            quat = [bbox[f'BoundingBox3D Orientation Quat {ax}'] for ax in ['X', 'Y', 'Z', 'W']]
            rot = Rotation.from_quat(quat).as_matrix()
            # rot = Rotation.from_euler('xyz', Rotation.from_quat(quat).as_euler('xyz', degrees=False), degrees=False).as_matrix()

            bbox_corners = (rot @ (corners * dims).T + center[:, None]).T
            bbox_corners_image = self.body_to_image(bbox_corners, intrinsic, extrinsic).astype(np.int32)

            color = (255, 255, 255)
            thickness = 1

            img = self.plot_annotations(img, bbox_corners, camera_params, extrinsic, color, thickness)
            if not fisheye:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner_of_text = (bbox_corners_image[5, 0], bbox_corners_image[5, 1])
                font_scale = 0.5 if img.shape[1] >= 1000 else 0.3
                thickness = 2 if img.shape[1] >= 1000 else 1

                cv2.putText(img, bbox['ObjectType'], bottom_left_corner_of_text, font, font_scale, color, thickness)

        return img

    def plot_annotations(self, img: np.array, bbox_corners: np.array, camera_params: CameraParams,
                         extrinsic: np.array, color: Color, thickness: int) -> np.array:
        """
        Projects annotations into images captured by a fisheye camera.

        Args:
            bbox_corners: 3D corner points in body coordinate system
            camera_params: camera parameters needed for projection
            color: visualization color
            extrinsic: body to sensor matrix, shape: [4, 4]
            img: camera image where annotations are visualized
            thickness: line thickness

        Returns:
            img: image with visualized annotations
        """
        bbox_corners_cam = self.body_to_camera(bbox_corners, extrinsic).T
        lines = []
        front = [np.array([bbox_corners_cam[i], bbox_corners_cam[i + 1]]) for i in range(3)]
        front.append(np.array([bbox_corners_cam[3], bbox_corners_cam[0]]))
        lines.append(front)

        back = [np.array([bbox_corners_cam[i], bbox_corners_cam[i + 1]]) for i in range(4, 7)]
        back.append(np.array([bbox_corners_cam[7], bbox_corners_cam[4]]))
        lines.append(back)

        between = [np.array([bbox_corners_cam[i], bbox_corners_cam[i + 4]]) for i in range(4)]
        lines.append(between)
        lines = np.concatenate(lines, axis=0).squeeze()

        img = self.plot_lines_to_image(img, lines, camera_params, color=color, thickness=thickness)

        return img

    def body_to_image(self, p: np.array, intrinsic: np.array, extrinsic: np.array) -> np.array:
        """
        Projects a 3D point in body coordinate system into image coordinate system.

        Args:
            p: 3D points in body coordinate system, shape: [N, 3]
            intrinsic: intrinsic calibration matrix, shape: [3, 4]
            extrinsic: body to sensor matrix, shape: [4, 4]

        Returns:
             p_im: projected points in image space, shape: [N, 2]
        """
        p_hom = np.concatenate([p, np.ones((p.shape[0], 1))], 1)
        p_im_hom = (intrinsic @ (extrinsic @ p_hom.T))
        p_im = (p_im_hom[:2] / p_im_hom[2]).T

        return p_im

    def body_to_camera(self, p: np.array, extrinsic: np.array) -> np.array:
        """
        Transforms a 3D point in body coordinate into camera coordinate system.

        Args:
            p: 3D points in body coordinate system, shape: [N, 3]
            extrinsic: body to sensor matrix, shape: [4, 4]

        Returns:
            p_cam: 3D points in camera coordinate system, shape: [3, N]
        """
        p_hom = np.concatenate([p, np.ones((p.shape[0], 1))], 1)
        p_cam_hom = extrinsic @ p_hom.T

        return p_cam_hom[:3, :]

    def plot_lines_to_image(self, img: np.array, lines: np.array, camera_params: CameraParams,
                            color: Color = (255, 0, 0), thickness: int = 1) -> np.array:
        """
        Args:
            img: camera image where annotations are visualized
            lines: shape: [N, 2, 3] (N: number of lines, 2 points per line, xyz coordinate for each point)
            camera_params: camera parameters needed for projection
            color: visualization color
            thickness: line thickness

        Returns:
            img: image with visualizations
        """
        lines = lines.reshape(-1, 2, 3)
        w, h = img.shape[1], img.shape[0]

        line_segment = 0.05
        all_lines = []
        for line in lines:
            pt0, pt1 = line
            length = np.linalg.norm(pt1 - pt0)
            slice_cnt = int(np.ceil(length / line_segment))
            line_pts = np.linspace(pt0, pt1, slice_cnt)
            line_segments = np.repeat(line_pts, 2, axis=0)[1:-1]
            all_lines.append(line_segments)
        all_lines = np.vstack(all_lines).reshape(-1, 2, 3)

        if camera_params.camera_model == 'mei':
            projected_lines = self.mei_camera_to_image(ray=all_lines.T, camera_params=camera_params).T
            projected_lines = projected_lines.reshape(-1, 4).astype(np.int)
            # Filter out of bounds lines.
            projected_lines = projected_lines[
                np.max(projected_lines >= 0, axis=1) & (projected_lines[:, 0] < w) | (projected_lines[:, 1] < h) | (
                        projected_lines[:, 2] < w) | (projected_lines[:, 3] < h)]
            projected_lines = projected_lines.reshape(-1, 2, 2).astype(np.int)
        else:
            projected_lines = self.pinhole_camera_to_image(ray=all_lines.T, camera_params=camera_params).T

            in_viewport_mask = np.min(projected_lines >= 0, axis=(1, 2)) & np.min(projected_lines < (w, h), axis=(1, 2))
            projected_lines = projected_lines[in_viewport_mask]

            unprojected_lines = self.image_to_pinhole_camera(image_points=projected_lines.T, camera_params=camera_params).T
            all_lines_in_viewport = all_lines[in_viewport_mask]

            def normalized(a, axis=-1, order=2):
                l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
                l2[l2 == 0] = 1
                return a / np.expand_dims(l2, axis)

            dot = np.sum(normalized(all_lines_in_viewport) * normalized(unprojected_lines), axis=-1)
            # dot close to 1 means same direction
            line_projectable = np.min(dot, axis=-1) > 0.999
            projected_lines = projected_lines[line_projectable]

        cv2.polylines(img, projected_lines.astype(np.int), isClosed=False, color=color, thickness=thickness)

        return img

    def is_in_fov(self, obj: Dict, sensor_name: str) -> bool:
        """
        Determines whether an object is in the field-of-view of a sensor.
        Very simple but fast logic, might be more sophisticated.

        Args:
            obj: an annotated object represented as a dict
            sensor_name: name of the sensor

        Returns:
            bool, whether the object is in the FOV of the sensor
        """
        x = obj['BoundingBox3D Origin X']
        y = obj['BoundingBox3D Origin Y']

        if 'right' in sensor_name:
            return y < -0.1

        if 'left' in sensor_name:
            return y > 0.1

        if 'front' in sensor_name:
            return x > 0.5

        if 'back' in sensor_name:
            return x < -0.5

    def mei_camera_to_image(self, ray: np.array, camera_params: CameraParams) -> np.array:
        """
        Projects points from mei camera space into image space.

        Args:
            ray: [3, 2, N] shaped array (N: number of lines, 2 points per line, xyz coordinate for each point)
            camera_params: camera parameters needed for projection

        Returns:
            [2, 2, N] shaped array (2 points per line, uv coordinate for each point, N: number of lines)
        """
        xi = camera_params.xi
        k1 = camera_params.dist_coeffs[0]
        k2 = camera_params.dist_coeffs[1]
        p1 = camera_params.dist_coeffs[2]
        p2 = camera_params.dist_coeffs[3]
        if len(camera_params.dist_coeffs) == 5:
            k3 = camera_params.dist_coeffs[4]
        else:
            k3 = 0

        principal_point_x, principal_point_y = camera_params.principal_point[0], camera_params.principal_point[1]
        focal_length_x, focal_length_y = camera_params.focal_length[0], camera_params.focal_length[1]

        ray_x, ray_y, ray_z = ray
        norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
        x = ray_x / norm
        y = ray_y / norm
        z = ray_z / norm + xi
        z[np.abs(z) <= 1e-5] = 1e-5

        x = x / z
        y = y / z

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        q_x = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        q_y = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
        q_x = q_x * focal_length_x + principal_point_x
        q_y = q_y * focal_length_y + principal_point_y

        return np.stack([q_x, q_y], axis=0)

    def pinhole_camera_to_image(self, ray: np.array, camera_params: CameraParams) -> np.array:
        """
        Projects points from pinhole camera space into image space.

        Args:
            ray: [3, 2, N] shaped array (N: number of lines, 2 points per line, xyz coordinate for each point)
            camera_params: camera parameters needed for projection

        Returns:
            [2, 2, N] shaped array (2 points per line, uv coordinate for each point, N: number of lines)
        """
        ray_x, ray_y, ray_z = ray
        x = ray_x / ray_z
        y = ray_y / ray_z

        def distortPoint(x, y):
            if not camera_params.dist_coeffs is None:
                k1 = camera_params.dist_coeffs[0]
                k2 = camera_params.dist_coeffs[1]
                p1 = camera_params.dist_coeffs[2]
                p2 = camera_params.dist_coeffs[3]
                k3 = camera_params.dist_coeffs[4]

                r2 = x * x + y * y
                r4 = r2 * r2
                r6 = r4 * r2

                coefficient = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

                qx = x * coefficient + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
                qy = y * coefficient + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
            else:
                qx = x
                qy = y
            return qx, qy

        qx, qy = distortPoint(x, y)
        mask = ((x < -5e-2) & (qx > 1e-5)) | ((x > 5e-2) & (qx < -1e-5)) | \
               ((y < -5e-2) & (qy > 1e-5)) | ((y > 5e-2) & (qy < -1e-5))

        delta = 2e-2
        x2 = x + delta * x
        y2 = y + delta * y
        q2x, q2y = distortPoint(x2, y2)

        mask |= (qx * q2x + qy * q2y) < 0
        mask |= (qx * qx + qy * qy) > (q2x * q2x + q2y * q2y)

        qx = qx * camera_params.focal_length[0] + camera_params.principal_point[0]
        qy = qy * camera_params.focal_length[1] + camera_params.principal_point[1]

        # viewport check
        mask |= (qx < 0.0) | (camera_params.principal_point[0] * 2 < qx) | \
                (qy < 0.0) | (camera_params.principal_point[1] * 2 < qy)

        mask = np.invert(mask)

        # for invalid points we get (-1.0, -1.0) values
        image_point = np.full((2,) + ray.shape[1:3], -1.0)
        image_point[0, mask] = qx[mask]
        image_point[1, mask] = qy[mask]

        return image_point

    def image_to_pinhole_camera(self, image_points: np.array, camera_params: CameraParams) -> np.array:
        """
        Projects points from image space into pinhole camera space.

        Args:
            image_points: [2, 2, N] shaped array (N: number of lines, 2 points per line, uv coordinate for each point)
            camera_params: camera parameters needed for projection

        Returns:
            [3, 2, N] shaped array (3 points per line, xyz coordinate for each point, N: number of lines)
        """
        distorted_image_point = np.zeros(image_points.shape)
        distorted_image_point[0] = (image_points[0] - camera_params.principal_point[0]) / camera_params.focal_length[0]
        distorted_image_point[1] = (image_points[1] - camera_params.principal_point[1]) / camera_params.focal_length[1]

        k1 = k2 = p1 = p2 = k3 = 0.0
        if camera_params.dist_coeffs is not None:
            k1 = camera_params.dist_coeffs[0]
            k2 = camera_params.dist_coeffs[1]
            p1 = camera_params.dist_coeffs[2]
            p2 = camera_params.dist_coeffs[3]
            k3 = camera_params.dist_coeffs[4]

        undistorted_image_point = np.copy(distorted_image_point)
        undistorted = k1 == 0.0 and k2 == 0.0 and p1 == 0.0 and p2 == 0.0 and k3 == 0.0
        if not undistorted:
            n_iterations = 20
            for i in range(n_iterations):
                xx = undistorted_image_point[0] * undistorted_image_point[0]
                yy = undistorted_image_point[1] * undistorted_image_point[1]
                r2 = xx + yy
                _2xy = 2.0 * undistorted_image_point[0] * undistorted_image_point[1]
                radial_distortion = 1.0 + (k1 + (k2 + k3 * r2) * r2) * r2
                tangential_distortion_X = np.array(p1 * _2xy + p2 * (r2 + 2.0 * xx))
                tangential_distortion_Y = np.array(p1 * (r2 + 2.0 * yy) + p2 * _2xy)

                undistorted_image_point[0] = (distorted_image_point[0] - tangential_distortion_X) / radial_distortion
                undistorted_image_point[1] = (distorted_image_point[1] - tangential_distortion_Y) / radial_distortion

        norm = np.sqrt(undistorted_image_point[0] * undistorted_image_point[0] + undistorted_image_point[1] *
                       undistorted_image_point[1] + 1.0)
        ray = np.zeros((3,) + image_points.shape[1:])
        ray[0] = undistorted_image_point[0] / norm
        ray[1] = undistorted_image_point[1] / norm
        ray[2] = 1.0 / norm

        return ray
