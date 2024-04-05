H = 704
W = 1280
final_dim = (704, 1280)


data_root = '/h/aimotive_dataset'
eval_split = 'night'
experiment_name = 'lidar_only_eval'
precision = 32
batch_size = 4
out_path = f'output/{experiment_name}'
log_wandb = False
num_workers = 8
learning_rate = 1e-3 / 64 * batch_size

voxel_size = [0.2, 0.2, 8]
out_size_factor = 4
point_cloud_range = [4*-51.2, -51.2/2, -5, 4*51.2, 51.2/2, 3]

use_cam   = False
use_lidar = True
use_radar = False
use_depth_loss = False
train_velocity = False
look_back    = 0
look_forward = 0
ckpt_path = 'exps/weights/lidar_only.ckpt/'

trainer_params = dict(enable_progress_bar=True,
                      precision=32, # 16 does not work yet
                      devices=1,
                      )


lidar_input_channels = 8 if use_radar else 5
lidar_feature_channels = 256 if use_lidar else 0
camera_feature_channels = 80 if use_cam else 0
fuse_layer_in_channels = camera_feature_channels + lidar_feature_channels

out_shape = [int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
             int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])]

backbone_conf = {
    'x_bound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0] * out_size_factor],
    'y_bound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1] * out_size_factor],
    'z_bound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    'd_bound': [2.0, point_cloud_range[3] + 1.6, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    camera_feature_channels,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}

ida_aug_conf = {
    'resize_lim': (1.0, 1.0),
    'final_dim':
    final_dim,
    'rot_lim': (-0.0, 0.0),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-5.0, 5.0),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}


bev_backbone = dict(
    type='ResNet',
    in_channels=fuse_layer_in_channels,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
    in_channels=[160, 320, 640],
    upsample_strides=[8, 16, 32],
    out_channels=[64, 64, 64])


CLASSES = [
    'car',
    'truck/bus',
    'motorcycle',
    'pedestrian',
    'other'
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=1, class_names=['truck/bus']),
    dict(num_class=1, class_names=['motorcycle']),
    dict(num_class=1, class_names=['pedestrian'])
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[point_cloud_range[0]-10.0, point_cloud_range[1]-10.0, -10, point_cloud_range[3]+10.0, point_cloud_range[4]+10.0, 10],
    max_num=500,
    score_threshold=0.0,
    out_size_factor=out_size_factor,
    voxel_size=voxel_size,
    pc_range=point_cloud_range,
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=point_cloud_range,
    grid_size=[out_shape[1], out_shape[0], 1],
    voxel_size=voxel_size,
    out_size_factor=out_size_factor,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3 if train_velocity else 0.0, 0.3 if train_velocity else 0.0],
)

test_cfg = dict(
    post_center_limit_range=point_cloud_range,
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 10, 0.5, 0.25],
    score_threshold=0.1,
    out_size_factor=out_size_factor,
    voxel_size=voxel_size,
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 192, # 512,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


lidar_conf = dict(
    type='MVXFasterRCNN',
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=15, voxel_size=voxel_size, max_voxels=(25000, 25000)
    ),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41] + out_shape,
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
    )
)
