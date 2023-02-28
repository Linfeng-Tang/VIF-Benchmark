#coding:utf-8
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""
import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np

def prepare_data_path(dataset_path):

    filenames = os.listdir(dataset_path)
    data_dir = os.path.join(os.getcwd(), dataset_path)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir,"*.jpg"))))
    data.sort()
    return data


def imread(path, is_grayscale=True):
    if is_grayscale:
        # flatten=True
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


"""
读取相关文件并且创建数据集
"""
data_path_vis = prepare_data_path('./My_dataset/train_vis')
data_path_ir = prepare_data_path('./My_dataset/train_ir')


path_length=min(len(data_path_vis),len(data_path_ir))
# 裁剪的图片大小
sub_image_size=256
# 移动步长
stride=32

sub_image_sequence=[]
sub_image_merge=[]

for i in range(path_length):
    # 进行了归一化呀 这只是归一化到了-1到1
    image_vis=((imread(data_path_vis[i])) /255)
    image_ir=((imread(data_path_ir[i]) ) / 255)
    # 判断image shape 大小 其实如果是3 可以进行一下 cv2togray
    if len(image_vis.shape) == 3:
        h, w, _ = image_vis.shape
    else:
        h, w = image_vis.shape

    if len(image_ir.shape) == 3:
        h, w, _ = image_ir.shape
    else:
        h, w = image_ir.shape

    for x in range(0, h - sub_image_size + 1, stride):
        for y in range(0, w -sub_image_size + 1, stride):
            sub_image_vis = image_vis[x:x + sub_image_size, y:y + sub_image_size]
            sub_image_sequence.append(sub_image_vis)
            sub_image_ir = image_ir[x:x + sub_image_size, y:y + sub_image_size]
            sub_image_sequence.append(sub_image_ir)
            sub_image_merge.append(sub_image_sequence)
            sub_image_sequence=[]

arrdata = np.asarray(sub_image_merge)

savepath=('./train.h5')

print(arrdata.shape)

with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=arrdata)










