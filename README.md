# (ECCV 2026) Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints

<p align="center">
  	<a href="https://shawnricardo.github.io/">Lijun Guo</a><sup>*1</sup>,
  	<a href="https://maxwell-zhao.github.io/">Haoyu Zhao</a><sup>*2,3</sup>,
  	<a href="https://scholar.google.com/citations?user=fZuqWe0AAAAJ&hl=zh-CN&oi=ao">Xingyue Zhao</a><sup>4</sup>,
  	<a href="https://scholar.google.com/citations?user=Enj1HGIAAAAJ&hl=zh-CN">Rong Fu</a><sup>5</sup>,
    <a href="https://www.researchgate.net/scientific-contributions/Linghao-Zhuang-2305439714">Linghao Zhuang</a><sup>1</sup>
    <a href="https://kyonhuang.top/">Siteng Huang</a><sup>6</sup>
    <a href="https://zyliatzju.github.io/">Zhongyu Li</a><sup>2,3</sup> and
    <a href="https://jszy.whu.edu.cn/zouhua/zh_CN/index.htm">Hua Zou</a><sup>✉1</sup>
</p>



<p align="center">
  	<sup>1</sup>School of Computer Science, Wuhan University &nbsp;&nbsp;
  	<sup>2</sup>Hong Kong Embodied AI Lab &nbsp;&nbsp;
  	<sup>3</sup>The Chinese University of Hong Kong &nbsp;&nbsp;
    <sup>4</sup>Peking Union Medical College &nbsp;&nbsp;
  	<sup>5</sup>University of Macau &nbsp;&nbsp;
  	<sup>6</sup>Zhejiang University &nbsp;&nbsp;
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2603.11606">arXiv</a> |
  <a href="https://maxwell-zhao.github.io/Articulat3D/">Project Page</a> |
  <a href="https://github.com/ShawnRicardo/Articulat3D">Code</a> |
  <a href="https://huggingface.co/datasets/ShawnRicardo/Articulat3D-Sim">Data</a>
</p>
<p align="center">
    * Equal contribution &nbsp;&nbsp;
    ✉ Corresponding author
</p>


## Overall

This is the official repository of **ECCV 2026** paper: Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints. For more information, please visit our project page.

<p align="center">
  <img src="assets/pipeline.png" alt="Articulat3D teaser" width="100%">
</p>

## Installation

### Installing dependencies

1. Prepare the environment from the project root.

```bash
conda create -n articulat3d python=3.10
conda activate articulat3d

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 "xformers>=0.0.27" --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install -r TAPIP3D/requirements.txt
pip install -r requirements.txt
```

2. Compile pointops2

```bash
cd TAPIP3D/third_party/pointops2
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../..
```

3. Compile pointnet_lib

```bash
cd TAPIP3D/third_party/pointnet_lib
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../..
```

4. Compile MegaSAM if you add the optional MegaSAM third-party dependency for monocular video input.

```bash
cd TAPIP3D/third_party/megasam/base
LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH python setup.py install
cd ../../../..
```

### Downloading checkpoints

Download our TAPIP3D model checkpoint [here](https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth) to `TAPIP3D/checkpoints/tapip3d_final.pth`

If you want to run TAPIP3D on monocular videos, you need to prepare the following checkpoints manually to run MegaSAM:

1. Download the DepthAnything V1 checkpoint from [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth) and put it to `TAPIP3D/third_party/megasam/Depth-Anything/checkpoints/depth_anything_vitl14.pth`

2. Download the RAFT checkpoint from [here](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) and put it to `TAPIP3D/third_party/megasam/cvd_opt/raft-things.pth`

Additionally, the checkpoints of [MoGe](https://wangrc.site/MoGePage/) and [UniDepth](https://github.com/lpiccinelli-eth/UniDepth.git) will be downloaded automatically when running the demo. Please make sure your network connection is available.

## Run

### Data Preparation

Put each scene under a dataset root directory. The default training scripts use `data/Articulat3DSimECCV` as the dataset root and `StorageFurniture_45194` as the example scene.

```text
data/
`-- Articulat3DSimECCV/
    `-- StorageFurniture_45194/
        |-- images/
        |   |-- 000000.png
        |   |-- 000001.png
        |   |-- ...
        |   `-- 000749.png
        |-- depth/
        |   |-- 000000.png
        |   |-- 000001.png
        |   |-- ...
        |   `-- 000749.png
        |-- masks/
        |   |-- 000000.png
        |   |-- 000001.png
        |   |-- ...
        |   `-- 000749.png
        |-- camera_pose.json
        |-- intrinsics.json
        `-- gt/
            `-- mobility_v2.json
```

The expected file formats are:

- `images/*.png`: RGB or RGBA images.
- `depth/*.png`: depth maps. The scripts convert them to metric depth with `depth / depth_scale`; the default `depth_scale` is `6553.5`.
- `masks/*.png`: foreground masks aligned with `images/` and `depth/`.
- `camera_pose.json`: a dictionary keyed by frame id, such as `"000000"`, where each value is a 4x4 camera pose matrix.
- `intrinsics.json`: one 3x3 camera intrinsic matrix.
- `gt/mobility_v2.json`: mobility annotation used by `get_obj_prior.py` to generate the joint prior.

After preprocessing, the same scene directory will also contain:

```text
data/Articulat3DSimECCV/StorageFurniture_45194/
|-- joint_priori.json
|-- joint_details.json
|-- trajectory_tapip3d.npz
`-- trajectory_tapip3d_visualization.npz
```

### Preprocess

Run preprocessing before training. The first step prepares the object and joint prior files for a scene, and the second step runs TAPIP3D to extract 3D trajectories.

Set the scene path variables from the project root:

```bash
DATA_DIR=data/Articulat3DSimECCV
SCENE_NAME=StorageFurniture_45194
SCENE_PATH=${DATA_DIR}/${SCENE_NAME}
```

1. Generate the object/joint prior files with [get_obj_prior.py](preprocess/get_obj_prior.py).

```bash
python preprocess/get_obj_prior.py \
  --scene_path "$SCENE_PATH"
```

This step writes `joint_priori.json` under the scene directory. It also validates the static and dynamic frame ranges used by the scene.

2. Extract TAPIP3D tracks with [extract_tapip3d_track.py](preprocess/extract_tapip3d_track.py).

```bash
python preprocess/extract_tapip3d_track.py \
  --data_dir "$DATA_DIR" \
  --scene_name "$SCENE_NAME" \
  --tapip3d_dir TAPIP3D/
```

This step first runs TAPIP3D to produce the raw scene track file, then builds the motion prior outputs used by training:
`joint_details.json`, `trajectory_tapip3d.npz`, and `trajectory_tapip3d_visualization.npz`.

### Training

Run the two training stages in order from the project root.

1. Train Stage 1 with [run_train_articulat3d.py](scripts/run_train_articulat3d.py).

```bash
python scripts/run_train_articulat3d.py
```

This stage initializes the Gaussian scene and free motion bases from `trajectory_tapip3d_visualization.npz`, then saves the Stage 1 checkpoint under the configured `work_dir`.

2. Train Stage 2 with [run_train_articulat3d_stage2.py](scripts/run_train_articulat3d_stage2.py).

```bash
python scripts/run_train_articulat3d_stage2.py
```

This stage loads the Stage 1 checkpoint, initializes the articulated motion model, and continues optimization with joint-aware motion bases.

## Acknowledgement

This code used resources from [PARIS](https://github.com/3dlg-hcvc/paris), [Shape of Motion](https://github.com/vye16/shape-of-motion), [TAPIP3D](https://github.com/zbw001/TAPIP3D) and [VideoArtGS](https://github.com/YuLiu-LY/VideoArtGS). We thank the authors for open-sourcing their awesome projects.

## Citation

```
@article{guo2026articulat3d,
  title={Articulat3D: Reconstructing Articulated Digital Twins From Monocular Videos with Geometric and Motion Constraints},
  author={Guo, Lijun and Zhao, Haoyu and Zhao, Xingyue and Fu, Rong and Zhuang, Linghao and Huang, Siteng and Li, Zhongyu and Zou, Hua},
  journal={arXiv preprint arXiv:2603.11606},
  year={2026}
}
```
