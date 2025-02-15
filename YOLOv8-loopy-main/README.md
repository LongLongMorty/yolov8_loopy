# YOLOv8-loopy

![1713533874344](image/README/1713533874344.png)

## 1.环境配置

conda 安装

```
conda create -n yolov8-loopy python==3.10
conda activate yolov8-loopy
pip install ultralytics
```

安装 ultralytics 时会自动安装 pytorch 和 torchvision，但不是 GPU 版本，因此需要卸载后自己手动安装支持 GPU 版本。

```
pip uninstall torch, torchvision
```

根据自己的 cuda 版本安装对应的 torch 和 torchvision。
PyTorch 下载：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

TorchVision 下载：[download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)

下载好 whl 文件后再使用 pip 安装

## 2.训练

图像放在 dataset/images 下，划分为 train 和 val，标签放在 dataset/labels 下，和 images 一样划分

修改 yaml 文件

训练：

```
python train.py
```

模型保存在 runs/detect/train 下

## 3.推理

在 detcet.py 中设置好需要推理的文件和模型

```
python detect.py
```

输出文件在 runs/detect/predict 下
