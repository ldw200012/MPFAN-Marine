# <p align="center">MPFAN-Marine</p>

<b>M</b>ulti-<b>P</b>erspective <b>F</b>eature <b>A</b>ggregation <b>N</b>etwork <b>(MPFAN)</b> is a LiDAR point cloud-based 3D object re-identification model, powered by the aggregation of multiple feature vectors extracted using various sub-network structures.

## 🚀 BUILD REPOSITORY
1. Clone the MPFAN git repository (Main Workspace)
```
git clone https://github.com/ldw200012/MPFAN-Marine.git
```
2. Clone LAMTK git repository (Lamtk Library)
```
cd MPFAN-Marine/
git clone https://github.com/c7huang/lamtk
```
## ⚙️ ENVIRONMENT SETUP
1. Pull docker image (the image is with CUDA-11.3)
```
docker pull daldidan/mpfan:latest
```
2. Run docker container. You need to fix the parameters in run_docker.py
```
python3 tools/run_docker.py
```
3. (In docker container) Setup the dependencies
```
cd /mpfan
python setup.py develop --user

cd /mpfan/lamtk
pip install -e .
```

## 📥 GET PRETRAINED WEIGHTS
We provide you the pretrained weights for the following models: PointNet, PointNeXt, DGCNN, DeepGCN, Point Transformer, SPoTr, MPFAN.

| Model          | Trained Epoch | # Params | Download  |
| -------------- | ------------- | -------- | --------- |
| PointNeXt      | 500           | -        | [LINK](#) |
| DGCNN          | 500           | -        | [LINK](#) |
| DeepGCN        | 500           | -        | [LINK](#) |
| Point Transformer | 500        | -        | [LINK](#) |
| SPoTr          | 500           | -        | [LINK](#) |
| MPFAN          | 500           | -        | [LINK](https://drive.usercontent.google.com/download?id=1pGCarCGP6N-qt4nYr8WU7YqgYSuvEJUT) |

## 🎯 TRAIN
```
./tools/bash/train.sh {GPU-ID} {model_name} {dataset_name}
```

## 🧪 TEST
```
./tools/bash/test.sh {GPU-ID} {model_name} {checkpoint_name} {dataset_name}
```

## 🙏 ACKNOWLEDGEMENTS
Out repository is based on <a href="https://github.com/bentherien/point-cloud-reid.git">point-cloud-reid</a>, <a href="https://github.com/open-mmlab/mmdetection3d.git">mmdetection3d</a>, and <a href="https://github.com/guochengqian/openpoints.git">openpoints</a>.

<!-- ## CITE OUR WORK -->