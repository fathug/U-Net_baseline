# -*- coding:utf-8 -*-
# @Time : 2022/5/27 14:58
# @Author : 
# @Note :
from PIL import Image
from torchvision.utils import save_image


def Image_Resize(path):
    img_data = Image.open(path)
    mask = Image.new('RGB', (max(img_data.size), max(img_data.size)))  # 新建方形掩膜
    mask.paste(img_data, (0, 0))
    mask.resize((256, 256))
    return mask

# #测试resize是否正常
# image_resized = Image_Resize('000012.jpg')
# image_resized.save('resized_image/resized_image.jpg')
