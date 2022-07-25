# -*- coding:utf-8 -*-
# @Time : 2022/5/27 14:41
# @Author : 
# @Note :1.制作数据集
from torch.utils.data import Dataset
import os
import utils
from torchvision import transforms

transforms = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):  # path为原图和标签图二者所在的文件夹
        self.path = path
        self.filenames = os.listdir(os.path.join(self.path, 'seg_image'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        # 索引图片
        seg_image_name = self.filenames[item]
        image_name = seg_image_name.replace('png', 'jpg')
        seg_image_path = os.path.join(self.path, 'seg_image', seg_image_name)
        image_path = os.path.join(self.path, 'image', image_name)
        # 对图片进行预处理，缩放、归一化...
        seg_image_resized = transforms(utils.Image_Resize(seg_image_path))
        image_resized = transforms(utils.Image_Resize(image_path))

        return seg_image_resized, image_resized
