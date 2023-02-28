# CUFD
Code of [CUFD: An encoder-decoder network for visible and infrared image fusion based on common and unique feature decomposition](https://www.sciencedirect.com/science/article/pii/S1077314222000352)

Tips
---------
#### To train:<br>
* Step1: Download [training dataset](https://pan.baidu.com/s/1yKJZu15aeSzqEjMjn-6Kag?pwd=0v11) or create your own training dataset by [code](https://github.com/hanna-xu/utils).
* Step2: In main.py, keep `IS_TRAINING==True` and choose the function train_part1.py (the 29th line in main.py), and then run main.py.
* Step3: In main.py, keep `IS_TRAINING==True` and choose the function train_part2.py (the 31th line in main.py), and then run main.py.

#### To test with the pre-trained model:<br>
* In main.py, keep `IS_TRAINING==False`, and run main.py.

If this work is helpful to you, please cite it as:
```
@article{xu2022cufd,
  title={CUFD: An encoder--decoder network for visible and infrared image fusion based on common and unique feature decomposition},
  author={Xu, Han and Gong, Meiqi and Tian, Xin and Huang, Jun and Ma, Jiayi},
  journal={Computer Vision and Image Understanding},
  pages={103407},
  year={2022},
  publisher={Elsevier}
}
```
If you have any question, please email to me (meiqigong@whu.edu.cn).
