# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
import numpy
import scipy.io as sio
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from PIL import Image
import datetime
import multiprocessing
import warnings
# from model.model import My_model
warnings.filterwarnings("ignore")


class MyDataSet(Dataset):
    def __init__(self,Data_root_path,data_inf,satellite_transform,bsvi_transform,train_flag = False,branch = 'all'):
        self.data_root_path = Data_root_path
        self.data_inf = data_inf
        self.train_flag = train_flag
        self.satellite_transform = satellite_transform
        self.bsvi_transform = bsvi_transform
        self.stallite_img_file = os.path.join(self.data_root_path,'Satellite')
        self.bsvi_sequence_file = os.path.join(self.data_root_path,'BSVI_IMGS')
        self.branch = branch

    def __getitem__(self, index):
        """
        迭代获得每一个样本，包括：场景图、场景序列和标签
        :param index:
        :return:
        """
        data_inf = self.data_inf[index]
        satellite_img_path = os.path.join(self.stallite_img_file,data_inf['satellite'])
        bsvi_sequence_path_list = [ os.path.join(self.bsvi_sequence_file, img_name) for img_name in data_inf['Scene_seq'] ]

        if self.branch == 'satellite':
            Sence_img = Image.open(satellite_img_path)
            Sence_img = self.satellite_transform(Sence_img)
            label = data_inf['label']
            return Sence_img,label
        if self.branch == 'bsvi':
            frame_sequence_list = []
            for img_path in bsvi_sequence_path_list:
                sequence_img = Image.open(img_path)
                sequence_img = self.bsvi_transform(sequence_img)
                frame_sequence_list.append(sequence_img)
            frame_sequence = torch.stack(frame_sequence_list)
            label = data_inf['label']
            return frame_sequence,label
        if self.branch == 'all':
            Sence_img = Image.open(satellite_img_path)
            Sence_img = self.satellite_transform(Sence_img)
            frame_sequence_list = []
            for img_path in bsvi_sequence_path_list:
                sequence_img = Image.open(img_path)
                sequence_img = self.bsvi_transform(sequence_img)
                frame_sequence_list.append(sequence_img)
            frame_sequence = torch.stack(frame_sequence_list)
            label= data_inf['label']
            return Sence_img,frame_sequence,label

    def __len__(self):
        return len(self.data_inf)

if __name__ == '__main__':
    Data_root_path = '../Dataset/'
    train_inf = [json.loads(line) for line in open(Data_root_path + 'train.json')][0]

    """
    图像预处理
    """
    # data_transform = transforms.Compose([transforms.Resize([192,384]),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transform = {'satellite':transforms.Compose([transforms.Resize([256, 512]), transforms.ToTensor()]),
                      'bsvi':transforms.Compose([transforms.Resize([256, 512]), transforms.ToTensor()])}

    train_loader = DataLoader(
        MyDataSet(Data_root_path, train_inf,satellite_transform = data_transform['satellite'],
                  bsvi_transform=data_transform['bsvi'], train_flag=True,branch='bsvi'),
        batch_size=16, shuffle=True,
        num_workers=0,
        pin_memory=True)


    # model = My_model().to('cuda')
    start_time = datetime.datetime.now()
    for epochs in range(1):
        for data in tqdm(train_loader, total=len(train_loader)):
            # pass
            # Sence_img,Sence_sequence,Label = data
            # Sence_img,Sence_sequence,Label = Sence_img.to('cuda'), Sence_sequence.to('cuda'),Label[0].to('cuda')
            Sence_img,Label = data
            Sence_img,Label = Sence_img.to('cuda'),Label[0].to('cuda')
            # output_accident, output_overspeed = model(Sence_img.to('cuda'),Sence_sequence.to('cuda'))
            # Scene_img = Sence_sequence[0,0, :, :, :]
            # Scene_img = Scene_img.cpu().numpy()
            # Scene_img = Scene_img.transpose(1, 2, 0)
            # plt.imshow(Scene_img)
            # plt.show()
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print('使用For循环的耗时为{}'.format(elapsed_time.total_seconds()))
    # print('使用Dali的耗时为{}'.format(elapsed_time.total_seconds()))