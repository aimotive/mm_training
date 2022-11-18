## Multimodal training on aiMotive Multimodal Dataset
This repository implements multimodal models trained on aiMotive Multimodal Dataset. For more details, please refer to our [paper on Arxiv](https://arxiv.org/abs/XXXX).
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
The config parameters are contained by exps/conf_aim.py. You need to override its values.

**Train.**
```
PYTHONPATH=$PYTHONPATH: python exps/mm_training_aim.py
```
**Eval.**
```
PYTHONPATH=$PYTHONPATH: python exps/eval.py
```

## Cite our work
If you use this code or aiMotive Multimodal Dataset in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{matuszka2022aimotivedataset,
  title = {aiMotive Dataset: A Multimodal Dataset for Robust Autonomous Driving with Long-Range Perception},
  author = {Matuszka, Tamás and Barton, Iván and Butykai, Ádám and Hajas, Péter and Kiss, Dávid and Kovács, Domonkos and Kunsági-Máté, Sándor and Lengyel, Péter and Németh, Gábor and Pető, Levente and Ribli, Dezső and Szeghy, Dávid and Vajna, Szabolcs and Varga, Bálint},
  doi = {10.48550/ARXIV.2211.09445},
  url = {https://arxiv.org/abs/2211.09445},
  publisher = {arXiv},
  year = {2022},
}
```
