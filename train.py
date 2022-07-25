# -*- coding:utf-8 -*-
# @Time : 2022/5/28 16:42
# @Author : 
# @Note :
import os.path

from utils import *
from data import *
from net import *
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

path_dataset = 'Dataset'
device = torch.device('cuda')
path_weight = 'param/unet_weight.pth'

if __name__ == '__main__':
    mydataset = MyDataset(path_dataset)
    mydatasetloader = DataLoader(mydataset, batch_size=2, shuffle=True, num_workers=1)

    unet = UNet()
    unet = unet.to(device)

    # 权重的加载--------------------------------
    if os.path.exists('param/unet_weight.pth'):
        unet.load_state_dict(torch.load('param/unet_weight.pth'))
        print('load successfully')
    else:
        print('no net weight')
    # ----------------------------------------

    for epoch in range(10):
        for i, (image, seg_image) in enumerate(mydatasetloader):
            image, seg_image = image.to(device), seg_image.to(device)
            out_image = unet(image)

            train_loss = nn.BCELoss()(out_image, seg_image)  # 计算损失函数
            optim.Adam(unet.parameters()).zero_grad()  # 清空梯度
            train_loss.backward()  # 反向计算
            optim.Adam(unet.parameters()).step()  # 更新梯度

        # 保存权重
        torch.save(unet.state_dict(), path_weight)
        print(f'epoch{epoch} loss={train_loss}')
