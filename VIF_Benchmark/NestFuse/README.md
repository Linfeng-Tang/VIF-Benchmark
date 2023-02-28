# NestFuse: An Infrared and Visible Image Fusion Architecture based on Nest Connection and Spatial/Channel Attention Models

[Hui Li](https://hli1221.github.io/), Xiao-Jun Wu*, Tariq S. Durrani  
IEEE Transactions on Instrumentation and Measurement, Volume: 69, issue: 12, pp. 9645–9656, Dec. 2020.  
- [IEEEXplore](https://ieeexplore.ieee.org/document/9127964) 
- [arXiv](https://arxiv.org/abs/2007.00328)

## Platform
Python 3.7  
Pytorch >=0.4.1  

## Analysis metrics

In "analysis_metrics_in_paper", seven metrics are included which were utilized in our paper.


## Fusion framework

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) (T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our auto-encoder network.

<img src="https://github.com/hli1221/imagefusion-nestfuse/blob/master/figures/framework_test-01.png" width="600">


## Fusion strategy (two attention models)  
In our fusion strategy, we focus on two types of features: spatial attention model and channel attention model. The extracted multi-scale deep features are processed in two phases.

<img src="https://github.com/hli1221/imagefusion-nestfuse/blob/master/figures/fusion_strategy_framework-01.png" width="600">


### Spatial attention model

<img src="https://github.com/hli1221/imagefusion-nestfuse/blob/master/figures/fusion_spatial-01.png" width="600">


### Channel attention model

<img src="https://github.com/hli1221/imagefusion-nestfuse/blob/master/figures/fusion_channel-01.png" width="600">



## NestFuse for RGBT visual object tracking
In this experiment, we choose SiamRPN++ \cite{li2019siamrpn++} as the base tracker and the fusion strategy proposed in this paper is applied to do the feature-level fusion. The SiamRPN++ is based on deep learning and achieves the state-of-the-art tracking performance in 2019.

![](https://github.com/hli1221/imagefusion-nestfuse/blob/master/figures/tracking_results-01.png)


If you have any question about this code, feel free to reach me(hui_li_jnu@163.com) 

# Citation

```
@article{li2020nestfuse,
 author = {Li, Hui and Wu, Xiao-Jun and Durrani, Tariq},
 title = {{NestFuse: An Infrared and Visible Image Fusion Architecture based on Nest Connection and Spatial/Channel Attention Models}},
 journal = {IEEE Transactions on Instrumentation and Measurement},
 volume={69},
 number={12}
 pages={9645--9656}
 year = {2020},
 note = {doi: 10.1109/TIM.2020.3005230},
 publisher={IEEE}
}
```


