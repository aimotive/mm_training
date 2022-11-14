import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import kornia
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data as tdata
import torch.utils.data.distributed
import torchvision
from mmdet3d.core import LiDARInstance3DBoxes
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from scipy.spatial.transform.rotation import Rotation
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

import wandb
from dataset.src.aimotive_dataset import AiMotiveDataset, collate_aim
from exps.conf_aim import *
from models.bev_depth import BEVDepthLiDAR
from utils.eval import MAPCalculator
from exps.mm_training_aim import BEVDepthLightningModel, parse_arguments, create_trainer

def main_infer():
    args = parse_arguments()
    model = BEVDepthLightningModel(**vars(args))
    trainer = create_trainer(args)

    trainer.predict(model, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main_infer()
