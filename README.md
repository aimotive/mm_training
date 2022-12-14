## Multimodal training on aiMotive Multimodal Dataset
This repository implements multimodal models trained on aiMotive Multimodal Dataset. For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/2211.09445).
The code is built on the top of [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) repository.

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).
```shell
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 1.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md)(v1.0.0rc4).


**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```

If you get an error message claiming 'No matching distribution found for waymo-open-dataset', then you can comment out
that line from mmdetection3d/requirements/optional.txt.

**Step 3.** Install mm_training (gpu required).
```shell
python setup.py develop
```

### Data preparation
**Step 1.** Download [aiMotive Multimodal Dataset](https://github.com/aimotive/aimotive_dataset).

**Step 2.** Change data_root variable in exps/conf_aim.py to the path of downloaded dataset.

### Tutorials
The config parameters are contained by exps/conf_aim.py. You need to override its values. Example config files can be found under exps/configs.

**Train.**
```
PYTHONPATH=$PYTHONPATH: python exps/mm_training_aim.py
```
**Eval.**
```
PYTHONPATH=$PYTHONPATH: python exps/eval.py
```

## Model checkpoints

| Model         | Checkpoint       |
| ------------- | ------------- |
| [LiDAR](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/ERYCINdt669NmVPOjNx_BNEBfOOCHCUJqU9l7QQscN6sYw?e=GWtchi)  | [link](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EfhqkK4yMMVKqmE9WRkuzUkBEPGLhQPpaPX9pt__KLfRZw?e=4euhQr)  |
| [LiDAR+cam](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/ET-4-eKDu5FCoG92npEBVlgBbVL-ckpBfxVPUYHsJrtgGQ?e=EpDLcE)  | [link](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EcU3gvbbbXVHgRvNA88D3RkB8-QegGGmzaQ03lbEVph8Bw?e=WPgd1E)  |
| [LiDAR+radar](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EQ5KKJBu4G9Irj-ImqP7mbABYs7Rdy2JxzLXgsHN9HuzJA?e=wNFqwl)  | [link](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EfYbJDNk3ddAlHfuwU6ebJEB1IM-TKCfWTYs9YEFIwTB5g?e=XyrUF8)  |
| [LiDAR+radar+cam](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EeHGiniCC8hMuciM9msEdIkB4RL0e2vgezrxSSiRlVKjQQ?e=s7lysC)  | [link](https://adasworks-my.sharepoint.com/:u:/g/personal/tamas_matuszka_aimotive_com1/EQlYxy4V15dAtv8bYpP7RrkBh4JXkbrtqN4egw4_RayODA?e=Fn7WfM)  |

## Cite our work
If you use this code or aiMotive Multimodal Dataset in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{matuszka2022aimotivedataset,
  title = {aiMotive Dataset: A Multimodal Dataset for Robust Autonomous Driving with Long-Range Perception},
  author = {Matuszka, Tam??s and Barton, Iv??n and Butykai, ??d??m and Hajas, P??ter and Kiss, D??vid and Kov??cs, Domonkos and Kuns??gi-M??t??, S??ndor and Lengyel, P??ter and N??meth, G??bor and Pet??, Levente and Ribli, Dezs?? and Szeghy, D??vid and Vajna, Szabolcs and Varga, B??lint},
  doi = {10.48550/ARXIV.2211.09445},
  url = {https://arxiv.org/abs/2211.09445},
  publisher = {arXiv},
  year = {2022},
}
```
