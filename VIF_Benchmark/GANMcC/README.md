# GANMcC
Code of paper GANMcC: A Generative Adversarial Network with Multi-classification Constraints for Infrared and Visible Image Fusion. In this paper, we propose a new fusion network, called generative adversarial network with multi-classification constraints (GANMcC), which transforms image fusion into a multi-distribution estimation problem. Experiments show that GANMcC can overcome the problem of unbalanced information fusion in previous methods, and the fused result not only has significant contrast but also contains rich texture details.
````
@article{ma2021ganmcc,
  title={GANMcC: A Generative Adversarial Network with Multi-classification Constraints for Infrared and Visible Image Fusion},
  author={Ma, Jiayi and Zhang, Hao and Shao, Zhenfeng and Liang, Pengwei and Xu, Han},
  journal={IEEE Transactions on Instrumentation and Measurement},
  volume={70},
  pages={5005014},
  year={2021},
  publisher={IEEE}
}
````




#### To train :<br>
Put training image pairs in the "Train_ir" and "Train_vi" folders, and run "CUDA_VISIBLE_DEVICES=0 python main.py" to train networks.


#### To test :<br>
Put test image pairs in the "Test_ir" and "Test_vi" folders, and run "CUDA_VISIBLE_DEVICES=0 python demo.py" to test the trained model.
You can also directly use the trained model we provide.


