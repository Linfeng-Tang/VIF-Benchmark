from __future__ import print_function
import time
import os
import tensorflow as tf
from scipy.misc import imread, imsave
from model import Model
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def img2RGB(f_name, vi_name):
    vi_img = Image.open(vi_name)
    vi_img = vi_img.convert('YCbCr')
    vi_Y, vi_Cb, vi_Cr = vi_img.split()
    f_img = Image.open(f_name).convert('L')
    f_img = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr))
    f_RGB = f_img.convert('RGB')
    f_RGB.save(f_name)


def main(Method = 'U2Fusion', model_path='/model/model.ckpt', ir_dir='', vi_dir='', save_dir='', is_RGB=True):  
    os.makedirs(save_dir, exist_ok=True)
    # fused_path = r'DenseFuse_Results_TNO'
    file_list = os.listdir(vi_dir)
    file_list = natsorted(file_list)
    
    M = Model(BATCH_SIZE=1, is_training=False)
    print(len(file_list))
    print('\nBegin to generate pictures ...\n')    
    test_bar = tqdm(file_list)
    
    with tf.Session() as sess:        
        t_list = tf.trainable_variables()
        saver = tf.train.Saver(var_list=t_list)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        for i, item in enumerate(test_bar):
            if item.endswith('.bmp') or item.endswith('.png') or item.endswith('.jpg') or item.endswith('.tif'):
                ir_image_name = os.path.join(os.path.abspath(ir_dir), item)
                vi_image_name = os.path.join(os.path.abspath(vi_dir), item)
                fused_image_name = os.path.join(os.path.abspath(save_dir), item)
                img1 = imread(ir_image_name, mode='L') / 255.0
                img2 = imread(vi_image_name, mode='L') / 255.0
                Shape1 = img1.shape
                h1 = Shape1[0]
                w1 = Shape1[1]
                Shape2 = img2.shape
                h2 = Shape2[0]
                w2 = Shape2[1]
                assert (h1 == h2 and w1 == w2), 'Two images must have the same shape!'
                img1 = img1.reshape([1, h1, w1, 1])
                img2 = img2.reshape([1, h1, w1, 1])
                start = time.time()
                outputs = sess.run(M.generated_img, feed_dict = {M.SOURCE1: img1, M.SOURCE2: img2})
                output = outputs[0, :, :, 0] # 0-1
                end = time.time()
                imsave(fused_image_name, output)
                if is_RGB:
                    img2RGB(fused_image_name, vi_image_name)
                test_bar.set_description('{} | {} | {:.4f} s'.format(Method, item, end-start))
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default=True, help='fusion results dir')
    parser.add_argument('--is_RGB', type=bool, default=True, help='colorize fused images with visible color channels')
    opts = parser.parse_args()
    main(
        Method=opts.Method, 
        model_path=opts.model_path,
        ir_dir = opts.ir_dir,
        vi_dir = opts.vi_dir,
        save_dir = opts.save_dir,
        is_RGB=opts.is_RGB
    )
