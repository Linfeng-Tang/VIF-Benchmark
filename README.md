# VIF_Benchmark
## 中文版
> 我们把所有主流的基于深度学习的红外和可见光图像融合方法都集成在了这个框架中。
> 
> 这些方法包括：[CSF](https://github.com/hanna-xu/CSF), [CUFD](https://github.com/Meiqi-Gong/CUFD), [DIDFuse](https://github.com/Meiqi-Gong/CUFD), [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion), [DenseFuse](https://github.com/hli1221/imagefusion_densefuse), [FusionGAN](https://github.com/jiayi-ma/FusionGAN), [GAN-FM](https://github.com/yuanjiteng/GAN-FM), [GANMcC](https://github.com/jiayi-ma/GANMcC), [IFCNN](https://github.com/uzeful/IFCNN), [NestFuse](https://github.com/hli1221/imagefusion-nestfuse), [PIAFusion](https://github.com/Linfeng-Tang/PIAFusion), [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020), [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest), [SDNet](https://github.com/HaoZhang1018/SDNet), [STDFusionNet](https://github.com/Linfeng-Tang/STDFusionNet), [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion), [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion), [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL), [U2Fusion](https://github.com/hanna-xu/U2Fusion), [UMF-CMGR](https://github.com/wdhudiekou/UMF-CMGR)

> 你可以把你的红外图像放在: './datsets/test_imgs/ir'文件夹下，可见光图像放在: './datsets/test_imgs/vi'文件夹下,

> 然后运行：**Python All_in_One.py**, 融合结果存放在 **./Results** 下。

> 如果你不需要要运行这么多方法，请修改**All_in_One.py**文件中的 **Method_list**。

> 为了确保程序能够正常运行请把这个项目在Ubuntu系统下面，并利用 **conda env create -f timer.yml** 安装相应的环境。

> 我的配置为： Ubuntu 18.04.3, TITAN RTX, CUDA 10.1,


> 整理不易，欢迎**Star**我们的项目，并引用我们以下的文献，你的支持是我们持续更新的动力。


```
@article{TangSeAFusion,
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

```
@article{TangSeAFusion,
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

```
@article{Tang2022SuperFusion,
  title={SuperFusion: A versatile image registration and fusion network with semantic awareness},
  author={Tang, Linfeng and Deng, Yuxin and Ma, Yong and Huang, Jun and Ma, Jiayi},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={9},
  number={12},
  pages={2121--2137},
  year={2022},
  publisher={IEEE}
}
```

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

```
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  volume = {83-84},
  pages = {79-92},
  year = {2022},
  issn = {1566-2535},
  publisher={Elsevier}
}
```

```
@article{Ma2021STDFusionNet,
  title={STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection},
  author={Jiayi Ma, Linfeng Tang, Meilong Xu, Hao Zhang, and Guobao Xiao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021},
  volume={70},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2021.3075747}，
  publisher={IEEE}
}
```

```
@article{Tang2022Survey,
  title={Deep learning-based image fusion: A survey},
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},  
  journal={Journal of Image and Graphics}
  volume={28},
  number={1},
  pages={3--36},
  year={2023}
}
```

> 由于Github的限制，目前DIVFusion的checkpoint缺少 **decom.ckpt**文件，你可以从作者的原始项目[DIVFusion](https://github.com/Linfeng-Tang/DIVFusion)中下载该checkpoint，也可联系我下载。

> 关于原始项目的问题，请根据对应项目的作者，如果有关于这个项目的问题，请联系：**linfeng0419@gmail.com** or **QQ：2458707789**（备注 姓名+学校）。由于项目的issue不会提醒，所以无法及时回复，请见谅。

> 部分融合结果展示如下：
> ![17.png in the TNO Dataset](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/17.png)
> ![17.png in the TNO Dataset](https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/00633D.png)
> 

https://github.com/Linfeng-Tang/VIF_Benchmark/blob/main/Demo/00633D.png


## English Version
> We have integrated all mainstream deep learning-based fusion methods for infrared and visible images into this framework. 

> These methods include:
[CSF](https://github.com/hanna-xu/CSF), [CUFD](https://github.com/Meiqi-Gong/CUFD), [DIDFuse](https://github.com/Meiqi-Gong/CUFD), [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion), [DenseFuse](https://github.com/hli1221/imagefusion_densefuse), [FusionGAN](https://github.com/jiayi-ma/FusionGAN), [GAN-FM](https://github.com/yuanjiteng/GAN-FM), [GANMcC](https://github.com/jiayi-ma/GANMcC), [IFCNN](https://github.com/uzeful/IFCNN), [NestFuse](https://github.com/hli1221/imagefusion-nestfuse), [PIAFusion](https://github.com/Linfeng-Tang/PIAFusion), [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020), [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest), [SDNet](https://github.com/HaoZhang1018/SDNet), [STDFusionNet](https://github.com/Linfeng-Tang/STDFusionNet), [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion), [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion), [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL), [U2Fusion](https://github.com/hanna-xu/U2Fusion), [UMF-CMGR](https://github.com/wdhudiekou/UMF-CMGR)

> You can place your infrared images in the './datsets/test_imgs/ir' folder and visible images in the './datsets/test_imgs/vi' folder, 

> Then run: Python All_in_One.py and the results will be saved in ./Results. 

> If you do not need to run so many methods, please modify the Method_list in the All_in_One.py file. 

> To ensure that the program runs correctly, please run this project on an Ubuntu system and install the corresponding environment using **conda env create -f timer.yml**. 
> 
> My configuration is: Ubuntu 18.04.3, TITAN RTX, CUDA 10.1.

> It is worth noting that due to Github's restrictions, the checkpoint for DIVFusion lacks the decom.ckpt file. You can download this checkpoint from the author's original project [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion), or contact me to obtain it.

> If you have any issues with the original projects, please contact the corresponding author. 

> If you have any issues with this project, please contact: linfeng0419@gmail.com or QQ: 2458707789 (please indicate your name and school in the remarks). Please note that due to the limitations of the issue system, we may not be able to reply promptly. 

> It's not easy to organize all of this, so please star our project and cite the following references. Your support is our driving force for continuous updates.

```
@article{TangSeAFusion,
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

```
@article{TangSeAFusion,
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

```
@article{Tang2022SuperFusion,
  title={SuperFusion: A versatile image registration and fusion network with semantic awareness},
  author={Tang, Linfeng and Deng, Yuxin and Ma, Yong and Huang, Jun and Ma, Jiayi},
  journal={IEEE/CAA Journal of Automatica Sinica},
  volume={9},
  number={12},
  pages={2121--2137},
  year={2022},
  publisher={IEEE}
}
```

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

```
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  volume = {83-84},
  pages = {79-92},
  year = {2022},
  issn = {1566-2535},
  publisher={Elsevier}
}
```

```
@article{Ma2021STDFusionNet,
  title={STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection},
  author={Jiayi Ma, Linfeng Tang, Meilong Xu, Hao Zhang, and Guobao Xiao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021},
  volume={70},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2021.3075747}，
  publisher={IEEE}
}
```

```
@article{Tang2022Survey,
  title={Deep learning-based image fusion: A survey},
  author={Tang, Linfeng and Zhang, Hao and Xu, Han and Ma, Jiayi},  
  journal={Journal of Image and Graphics}
  volume={28},
  number={1},
  pages={3--36},
  year={2023}
}
```





