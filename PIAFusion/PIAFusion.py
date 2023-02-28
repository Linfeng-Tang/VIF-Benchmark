# from train import model
from model import PIAFusion
import numpy as np
import tensorflow as tf
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
def main(Method = 'PIAFusion', model_path='', ir_dir='', vi_dir='', save_dir='', is_RGB=True): 
    parser = argparse.ArgumentParser()    
    parser.add_argument('--epoch', type=int, default=30, help='Number of epoch [10]')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch images [128]')
    parser.add_argument('--image_size', type=int, default=64, help='The size of image to use [33]')
    parser.add_argument('--label_size', type=int, default=2, help='The size of label to produce [21]')    
    parser.add_argument('--learning_rate', type=int, default=1e-3, help='The learning rate of gradient descent algorithm [1e-4]') 
    parser.add_argument('--stride', type=int, default=24, help='The size of stride to apply input image [14]')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Name of checkpoint directory [checkpoint]')
    parser.add_argument('--summary_dir', type=str, default='log', help='Name of log directory [log]')
    parser.add_argument('--model_type', type=str, default='PIAFusion', help='Illum for training the Illumination Aware network, PIAFusion for training the Fusion Network [PIAFusion]')
    parser.add_argument('--DataSet', type=str, default='MSRS', help='The Dataset for Testing, TNO, RoadScene, MSRS,  [TNO]')
    parser.add_argument('--is_train', type=bool, default=False, help='True for training, False for testing [True]') 
    FLAGS = parser.parse_args([])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        piafusion = PIAFusion(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      checkpoint_dir=model_path,
                      model_type=FLAGS.model_type,
                      phase=FLAGS.is_train,
                      Data_set=FLAGS.DataSet, 
                      Method=Method,
                      ir_dir=ir_dir,
                      vi_dir=vi_dir,
                      save_dir=save_dir)
        if FLAGS.is_train:
            piafusion.train(FLAGS)
        else:
            piafusion.test(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Method', type=str, default='TarDAL', help='Method name')
    parser.add_argument('--model_path', type=str, default='/data/timer/Comparison/VIF/TarDAL/weights/tardal++.pt', help='pretrained weights path')
    parser.add_argument('--ir_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/ir', help='infrared images dir')
    parser.add_argument('--vi_dir', type=str, default='/data/timer/Comparison/VIF/Dataset/test_imgs/vi', help='visible image dir')
    parser.add_argument('--save_dir', type=str, default='', help='fusion results dir')
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

