import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image
from natsort import natsorted

import os

class FuseDataVSM(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path, crop=lambda x: x):
        super(FuseDataVSM, self).__init__()
        self.crop = crop
        # gain infrared and visible images list
        self.ir_list = [x for x in natsorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in natsorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in natsorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in natsorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)


        return (ir, vi), (str(ir_path), str(vi_path)), (ir_map, vi_map)

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


class FuseTestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path):
        super(FuseTestData, self).__init__()
        # gain infrared and visible images list
        self.ir_list = natsorted(os.listdir(ir_folder))
        self.vi_list = natsorted(os.listdir(vi_folder))
        self.ir_folfder = ir_folder
        self.vi_folfder = vi_folder
        # self.ir_list = [x for x in natsorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        # print(file_list)
        # self.vi_list = [x for x in natsorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = os.path.join(self.ir_folfder, self.ir_list[index])
        vi_path = os.path.join(self.vi_folfder, self.vi_list[index])

        # assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        return (ir, vi), (str(ir_path), str(vi_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts


