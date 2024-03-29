# Non-residual Unrestricted Pruned Ultra-faster Line Detection
Ultra-faster Line Detection for lane detection and belt line detection.
The fastest lightweight lane line detection algorithm!
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
Code for training will be released in future.

# Trained models
We provide trained models on Tusimple at r=0.6. Other model will be released in future.

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
If you use the techniques in this article please cite:
```
@article{chen2023non,
  title={Non-Residual Unrestricted Pruned Ultra-faster Line Detection for Edge Devices},
  author={Chen, Pengpeng and Liu, Dongjingdian and Gao, Shouwan},
  journal={Pattern Recognition},
  pages={109321},
  year={2023},
  publisher={Elsevier}
}
```

