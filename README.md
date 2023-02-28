# VIF_Benchmark
## 中文版
> 我们把所有主流的基于深度学习的红外和可见光图像融合方法都集成在了这个框架中。
> 这些方法包括：[CSF](https://github.com/hanna-xu/CSF), [CUFD](https://github.com/Meiqi-Gong/CUFD), [DIDFuse](https://github.com/Meiqi-Gong/CUFD), [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion), [DenseFuse](https://github.com/hli1221/imagefusion_densefuse), [FusionGAN](https://github.com/jiayi-ma/FusionGAN), [GAN-FM](https://github.com/yuanjiteng/GAN-FM), [GANMcC](https://github.com/jiayi-ma/GANMcC), [IFCNN](https://github.com/uzeful/IFCNN), [NestFuse](https://github.com/hli1221/imagefusion-nestfuse), [PIAFusion](https://github.com/Linfeng-Tang/PIAFusion), [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020), [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest), [SDNet](https://github.com/HaoZhang1018/SDNet), [STDFusionNet](https://github.com/Linfeng-Tang/STDFusionNet), [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion), [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion), [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL), [U2Fusion](https://github.com/hanna-xu/U2Fusion), [UMF-CMGR](https://github.com/wdhudiekou/UMF-CMGR)
你可以把你的红外图像放在: './datsets/test_imgs/ir'文件夹下，可见光图像放在: './datsets/test_imgs/vi'文件夹下,
然后运行：**Python All_in_One.py**
如果你不需要要运行这么多方法，请修改**All_in_One.py**文件中的 **Method_list**, 结果存放在 **./Results** 下面。
为了确保程序能够正常运行请把这个项目在Ubuntu系统下面，并利用 **conda env create -f timer.yml** 安装相应的环境。
我的配置为： Ubuntu 18.04.3, TITAN RTX, CUDA 10.1,
整理不易，欢迎Star我们的项目，并引用我们以下的文献，你的支持是我们持续更新的动力。


由于Github的限制，目前DIVFusion的checkpoint缺少 **decom.ckpt**文件，你可以从作者的原始项目[DIVFusion](https://github.com/Linfeng-Tang/DIVFusion)中下载该checkpoint，也可联系我下载。
关于原始项目的问题，请根据对应项目的作者，如果有关于这个项目的问题，请联系：**linfeng0419@gmail.com** or **QQ：2458707789**（备注 姓名+学校）。由于项目的issue不会提醒，所以无法及时回复，请见谅。

## English Version
We have integrated all mainstream deep learning-based fusion methods for infrared and visible images into this framework. These methods include:
[CSF](https://github.com/hanna-xu/CSF), [CUFD](https://github.com/Meiqi-Gong/CUFD), [DIDFuse](https://github.com/Meiqi-Gong/CUFD), [DIVFusion](https://github.com/Linfeng-Tang/DIVFusion), [DenseFuse](https://github.com/hli1221/imagefusion_densefuse), [FusionGAN](https://github.com/jiayi-ma/FusionGAN), [GAN-FM](https://github.com/yuanjiteng/GAN-FM), [GANMcC](https://github.com/jiayi-ma/GANMcC), [IFCNN](https://github.com/uzeful/IFCNN), [NestFuse](https://github.com/hli1221/imagefusion-nestfuse), [PIAFusion](https://github.com/Linfeng-Tang/PIAFusion), [PMGI](https://github.com/HaoZhang1018/PMGI_AAAI2020), [RFN-Nest](https://github.com/hli1221/imagefusion-rfn-nest), [SDNet](https://github.com/HaoZhang1018/SDNet), [STDFusionNet](https://github.com/Linfeng-Tang/STDFusionNet), [SeAFusion](https://github.com/Linfeng-Tang/SeAFusion), [SuperFusion](https://github.com/Linfeng-Tang/SuperFusion), [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion), [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL), [U2Fusion](https://github.com/hanna-xu/U2Fusion), [UMF-CMGR](https://github.com/wdhudiekou/UMF-CMGR)
