# Non-residual Unrestricted Pruned Ultra-faster Line Detection
Ultra-faster Line Detection for lane detection and belt line detection.
Work step by step open.
# Demo 
<a href="https://www.bilibili.com/video/BV1Q44y1k7m4?share_source=copy_web
" target="_blank"><img src="https://github.com/vvhj/NRUPU/blob/main/images/20.png" 
alt="Demo" width="240" height="180" border="10" /></a>

# Install
Please see [INSTALL.md](./INSTALL.md)

# For evaluation, run
***
```Shell
python xxxx.py configs/xxx.py
```
deviation.py : deviation for beltline，
test.py : test accuarcy and speed.
# 自定义数据集
https://blog.csdn.net/qq_26894673/article/details/100052191

# Trained models
We provide trained Res-18 models on Tusimple.

|  Dataset  | Metric This repo | Avg FPS on GTX 2080Ti |    Avg FPS on NJNX    |    Model    |
|:--------: |:----------------:|:---------------------:|:---------------------:|:-----------:|
| Tusimple  |       95.62      |         682           |         115           | [GoogleDrive](https://drive.google.com/file/d/1zdlQ9IwzQqcbmqUEfWNjUzKPM-strawk/view?usp=sharing)|

# Thanks
Thanks (UFAST) https://github.com/cfzd/Ultra-Fast-Lane-Detection and (RepVGG) https://github.com/DingXiaoH/RepVGG. Please use following citation to show respect.

```BibTeX
@InProceedings{qin2020ultra,
author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
title = {Ultra Fast Structure-aware Deep Lane Detection},
booktitle = {The European Conference on Computer Vision (ECCV)},
year = {2020}
}

@inproceedings{ding2021repvgg,
title={Repvgg: Making vgg-style convnets great again},
author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={13733--13742},
year={2021}
}
```

