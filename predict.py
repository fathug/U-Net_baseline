# -*- coding:utf-8 -*-
# @Time : 2022/5/29 9:54
# @Author : 
# @Note :

import torch
from data import *
from net import *
import os
from utils import *

weight_path = 'param/unet_weight.pth'

unet = UNet()

# load net param
if os.path.exists(weight_path):
    unet.load_state_dict(torch.load(weight_path))
    print('load successfully')
else:
    print('no net param')

# 获取给定的图片
print('please input path of unpredict image:')
img0 = input()

img=Image_Resize(img0)
img=transforms(img)
img=img.unsqueeze(0)

img_out = unet(img)
save_image(img_out, 'result/image_pred.jpg')



