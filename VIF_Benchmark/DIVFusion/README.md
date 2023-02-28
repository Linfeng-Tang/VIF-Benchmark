# DIVFusion
DIVFusion: Darkness-free infrared and visible image fusion  (*^▽^*)

This is official Tensorflow implementation of "[DIVFusion: Darkness-free infrared and visible image fusion](https://www.sciencedirect.com/science/article/pii/S156625352200210X?via%3Dihub)"

## Framework
![The overall framework of the proposed DIVFusion algorithm.](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Framework.png)
The overall framework of the proposed DIVFusion. SIDNet is a network to seperate illumination degradation. TCEFNet integrates and enhances the complementary information of source images.

## Network Architecture1
![The architecture of the real-time infrared and visible image fusion network based on gradient residual dense block.](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/SIDNet.png)
The framework of the scene-illumination disentangled network (SIDNet).

## Network Architecture2
![The architecture of the real-time infrared and visible image fusion network based on gradient residual dense block.](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/TCEFNet.png)
The detailed structure of (a) gradient retention module (GRM) and (b) contrast block.

## Before Train
```**conda env create -f XXY_DIVFusion.yml**```

## To Train
Add VGG16.npy from to the file. Link:[here](https://pan.baidu.com/s/18OZqwjMXaIxPbHIzKZmlzA?pwd=1xo5), in which the extraction code is: **1xo5**.

First Run ```**CUDA_VISIBLE_DEVICES=0 python decomposition.py**``` to train your model(SIDNet).

Second Run ```**CUDA_VISIBLE_DEVICES=0 python fusion_enhancement_new.py**``` to train your model(TCEFNet).

The training data are selected from the LLVIP dataset. For convenient training, users can download the training dataset from [here](https://pan.baidu.com/s/1i5dIXJcus8_qy9Rq8DQWcw?pwd=he31), in which the extraction code is: **he31**.

The LLVIP dataset can be downloaded via the following link: [here](https://bupt-ai-cz.github.io/LLVIP/).

## To Test
The checkpoint can be found via the following link: [here](https://pan.baidu.com/s/1fcqkGc3F1fGsY8CaUcM4sA?pwd=dv3s), in which the extraction code is: **dv3s**.
The testing data are selected from the LLVIP dataset. link: [here](https://pan.baidu.com/s/1DA1VprRaV58n0ueB7C1Hbw?pwd=6hvf) The extraction code is: **6hvf**

Run ```**CUDA_VISIBLE_DEVICES=0 python test.py**``` to test the whole model.
## Demo
![Demo](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Example.png)
Nighttime infrared and visible image fusion results.

## Fusion Example
![Fusion Example](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Fusion.png)
Vision quality comparison of our method with seven SOTA fusion methods on #010064 and #060193 images from LLVIP dataset.

## Two-stage Fusion Example
![Two-stage Fusion Example](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Enhancement_fusion.png)
Vision quality comparison of two-stage fusion experiments. Each row represents a scene, and from top to bottom is #21006, #220312, and #260092 images from LLVIP
dataset. ((a)-(b): source images, (c)-(i): two-stage fusion results by different enhancement methods and fusion methods, (j): our fusion result).

## Generalization Example
![Generalization Example](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Fusion_MSRS.png)
Segmentation results for infrared, visible and fused images from the MFNet dataset. The segmentation model is Deeplabv3+, pre-trained on the Cityscapes dataset. Each
two rows represent a scene.

## Detection Results
![Detection Results](https://github.com/Xinyu-Xiang/DIVFusion/blob/main/Figure/Detection.png)
Object detection results for infrared, visible and fused images from the MFNet dataset. The YOLOv5 detector, pre-trained on the Coco dataset is deployed to achieve
object detection.


## If this work is helpful to you, please cite it as：
```
@article{Tang2022DIVFusion,
  title={DIVFusion: Darkness-free infrared and visible image fusion},
  author={Tang, Linfeng and Xiang, Xinyu and Zhang, Hao and Gong, Meiqi and Ma, Jiayi},
  journal={Information Fusion},
  volume = {91},
  pages = {477-493},
  year = {2023},
  publisher={Elsevier}
}
```
