# DIDFuse
**Codes for *"DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion"* (IJCAI 2020)**

- [*[Paper]*](https://www.ijcai.org/Proceedings/2020/135)
- [*[ArXiv]*](https://arxiv.org/abs/2003.09210v1)

## Citation

*[Zixiang Zhao](https://zhaozixiang1228.github.io/), [Shuang Xu](https://xsxjtu.github.io/), [Chunxia Zhang](https://scholar.google.com/citations?user=b5KG5awAAAAJ&hl=zh-CN) and [Junmin Liu](https://scholar.google.com/citations?user=C9lKEu8AAAAJ&hl=zh-CN), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), and [Pengfei Li](), DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion. IJCAI 2020: 970-976, https://www.ijcai.org/Proceedings/2020/135.*

```
@inproceedings{ZhaoDIDFuse2020,
  author    = {Zixiang Zhao and
               Shuang Xu and
               Chunxia Zhang and
               Junmin Liu and
               Jiangshe Zhang and
               Pengfei Li},
  title     = {DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion},
  booktitle = {{IJCAI}},
  pages     = {970--976},
  publisher = {ijcai.org},
  year      = {2020}
}
```

## Abstract

Infrared and visible image fusion, a hot topic in the field of image processing, aims at obtaining fused images keeping the advantages of source images. This paper proposes a novel auto-encoder (AE) based fusion network. The core idea is that the encoder decomposes an image into background and detail feature maps with low- and high-frequency information, respectively, and that the decoder recovers the original image. To this end, the loss function makes the background/detail feature maps of source images similar/dissimilar. In the test phase, background and detail feature maps are respectively merged via a fusion module, and the fused image is recovered by the decoder. Qualitative and quantitative results illustrate that our method can generate fusion images containing highlighted targets and abundant detail texture information with strong reproducibility and meanwhile surpass state-of-the-art (SOTA) approaches.

## Usage

### Training
A pretrained model is available in ```'./Models/Encoder_weight_IJCAI.pkl'``` and ```'./Models/Decoder_weight_IJCAI.pkl'```. We train it on FLIR (180 image pairs) in ```'./Datasets/Train_data_FLIR'```. In the training phase, all images are resize to 128x128 and are transformed to gray pictures.

If you want to re-train this net, you should run ```'train.py'```.

### Testing
The test images used in the paper have been stored in ```'./Test_result/TNO_IJCAI'```, ```'./Test_result/NIR_IJCAI'``` and ```'./Test_result/FLIR_IJCAI'```, respectively.

For other test images, run ```'test.py'``` and find the results in ```'./Test_result/'```.

## DIDFuse

### Illustration of our DIDFuse model.

<img src="image//Framework.png" width="100%" align=center />

### Qualitative fusion results.

<img src="image//Qualitative.png" width="90%" align=center />


### Quantitative  fusion results.

<img src="image//Quantitative.png" width="90%" align=center />

## Related Work

- Zixiang Zhao, Shuang Xu, Jiangshe Zhang, Chengyang Liang, Chunxia Zhang and Junmin Liu, "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2021.3075745, https://ieeexplore.ieee.org/document/9416456.

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang, *Bayesian fusion for infrared and visible images*, Signal Processing, Volume 177, 2020, 107734, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2020.107734.

