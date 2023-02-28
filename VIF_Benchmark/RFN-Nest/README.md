# RFN-Nest: An end-to-end residual fusion network for infrared and visible images

[Hui Li](https://hli1221.github.io/), Xiao-Jun Wu*, Josef Kittler  
Information Fusion (IF:13.669), Volume: 73, Pages: 72-86, September 2021  
[paper](https://doi.org/10.1016/j.inffus.2021.02.023)  
[arXiv](https://arxiv.org/abs/2103.04286)  
[Supplementary Material]()


## Platform
Python 3.7  
Pytorch >=0.4.1  

The testing datasets are included in "images".

The results iamges are included in "outputs".

## Training Dataset

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) (T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our auto-encoder network.

[KAIST](https://sites.google.com/view/multispectral/home) (S. Hwang, J. Park, N. Kim, Y. Choi, I. So Kweon, Multispectral pedestrian detection: Benchmark dataset and baseline, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1037–1045.) is utilized to train the RFN modules.

## Fusion framework

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/framework.png" width="600">


### Decoder architecture

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/decoder.png" width="600">



### Training RFN modules

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/training-rfn.png" width="600">


### Fusion results

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/results-umbrella.png" width="600">


## RFN for RGBT tracking - framework

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/tracking-framework.png" width="600">


### RFN for RGBT tracking - results

<img src="https://github.com/hli1221/imagefusion-rfn-nest/blob/main/framework/results-tracking.png" width="600">


If you have any question about this code, feel free to reach me(hui_li_jnu@163.com) 

# Citation

```
@article{li2021rfn,
  title={RFN-Nest: An end-to-end residual fusion network for infrared and visible images},
  author={Li, Hui and Wu, Xiao-Jun and Kittler, Josef},
  journal={Information Fusion},
  volume={73},
  pages={72--86},
  month={March},
  year={2021},
  publisher={Elsevier}
}
```

## A clerical error in our paper

I am very sorry about this clerical error. 
Actually, in Section 4.6, this part "With the nest connection, the decoder is able to preserve more image information conveyed by the multiscale deep features (𝑀𝐼, 𝐹 𝐹𝑀𝐼𝑑𝑐𝑡, 𝐹 𝐹𝑀𝐼𝑤) and generate more natural and clearer fused image (𝐸𝑁, 𝑆𝐷, 𝑉 𝐼𝐹)." should change to "With the nest connection, the decoder is able to preserve more image information conveyed by the multiscale deep features (𝑀𝐼, Nabf, MS-SSIM) and generate more natural and clearer fused image (𝐸𝑁, 𝑆𝐷, SCD)."

