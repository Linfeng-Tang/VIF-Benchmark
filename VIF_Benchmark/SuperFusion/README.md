


#  SuperFusion

This is official Pytorch implementation of "[SuperFusion: A Versatile Image Registration and Fusion Network with Semantic Awareness](https://www.sciencedirect.com/science/article/pii/S1566253521002542)"

## Framework
![The overall framework of the proposed SuperFusion for cross-modal image registration and fusion.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Overframe.jpg)
The overall framework of the proposed SuperFusion for cross-modal image registration and fusion.

## Network Architecture
### Dense Matcher
![The architecture of dense matcher, which consists of a pyramid feature extractor and iterative flow estimators. Flows are estimated in three scales iteratively and summed up.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/DenseMatcher.jpg)

The architecture of dense matcher, which consists of a pyramid feature extractor and iterative flow estimators. Flows are estimated in three scales iteratively and summed up.
### Fusion Network
![Architecture of the fusion network $\mathcal{N}_F$. Conv($c, k$) denotes a convolutional layer with $c$ output channels and kernel size of $k\times k$; GSAM indicates the global spatial attention module.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/FusionNet.jpg)
Architecture of the fusion network $\mathcal{N}_F$. Conv($c, k$) denotes a convolutional layer with $c$ output channels and kernel size of $k\times k$; GSAM indicates the global spatial attention module.

### Global Spatial Attention Module (GSAM)
![The schematic illustration of the global spatial attention module (GSAM). The global attention is calculated by adapting a spatial RNN to aggregate the spatial context in four directions.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/SAM.jpg)
The schematic illustration of the global spatial attention module (GSAM). The global attention is calculated by adapting a spatial RNN to aggregate the spatial context in four directions.

## Recommended Environment

 - [ ] torch  1.10.1 
 - [ ] torchvision 0.11.2 
 - [ ] kornia 0.6.5
 - [ ] opencv  4.5.5 
 - [ ] pillow  9.2.0
 
## To Train

## To Test





## Registration Results
Quantitative registration performance on MSRS and RoadScene. Mean reprojection error (RE) and end-point error (EPE) are reported.
![Quantitative registration performance on MSRS and RoadScene. Mean reprojection error~(RE) and end-point error~(EPE) are reported.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Reg-table.jpg)


![Qualitative registration performance of DASC, RIFT, GLU-Net, UMF-CMGR, CrossRAFT, and our SuperFusion. The first four rows of images are from the MSRS dataset, and the last two are from the RoadScene dataset. The purple textures are the gradients of registered infrared images and the backgrounds are the corresponding ground truths. The discriminateive regions that demonstrate the superiority of our method are highlighted in boxes. Note that, the gradients of the second column images are from the warped images, i.e. , the misaligned infrared images.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Reg.jpg)
Qualitative registration performance of DASC, RIFT, GLU-Net, UMF-CMGR, CrossRAFT, and our SuperFusion. The first four rows of images are from the MSRS dataset, and the last two are from the RoadScene dataset. The purple textures are the gradients of registered infrared images and the backgrounds are the corresponding ground truths. The discriminateive regions that demonstrate the superiority of our method are highlighted in boxes. Note that, the gradients of the second column images are from the warped images, i.e., the misaligned infrared images.

## Fusion Results
![Quantitative comparison results of SuperFusion with five state-of-the-art alternatives on the MSRS dataset.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/MSRS.jpg)

Quantitative comparison results of SuperFusion with five state-of-the-art alternatives on $361$ image pairs from the MSRS dataset. A point $(x, y)$ on the curve denotes that there are $100 * x$ percent of image pairs that have metric values no more than $y$.

![Quantitative comparison results of SuperFusion with five state-of-the-art alternatives on the RoadScene dataset.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/RoadScene.jpg)

Quantitative comparison results of SuperFusion with five state-of-the-art alternatives on $25$ image pairs from the RoadScene dataset.

![Qualitative comparison results of SuperFusion with five state-of-the-art infrared and visible image fusion methods on the MSRS and RoadScene datasets. All methods employ the built-in registration module (e.g., UMF-CMGR and our SuperFusion) or CrossRAFT to register the source images.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Fusion.jpg)
Qualitative comparison results of SuperFusion with five state-of-the-art infrared and visible image fusion methods on the MSRS and RoadScene datasets. All methods employ the built-in registration module (e.g., UMF-CMGR and our SuperFusion) or CrossRAFT to register the source images.


## Segmentation Results
Segmentation performance (IoU) of visible, infrared, and fused images on the MSRS dataset.
![Segmentation results for source images and fused images from the MSRS dataset.  The fused image indicates the fusion result generated by our SuperFusion, and the pre-trained segmentation model is provided by SeAFusion.](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Segmentation-table.jpg)


![Segmentation results for source images and fused images from the MSRS dataset. ](https://github.com/Linfeng-Tang/SuperFusion/blob/main/Figure/Segmentation.jpg)
Segmentation results for source images and fused images from the MSRS dataset.  The fused image indicates the fusion result generated by our SuperFusion, and the pre-trained segmentation model is provided by [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion).




## If this work is helpful to you, please cite it as：
```
@article{TANG2022SeAFusion,
title = {Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network},
author = {Linfeng Tang and Jiteng Yuan and Jiayi Ma},
journal = {Information Fusion},
volume = {82},
pages = {28-42},
year = {2022},
issn = {1566-2535},
publisher={Elsevier}
}
```
