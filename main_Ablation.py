# -*- coding:utf-8 -*-
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.Data_Load import MyDataSet
from utils.model.Ablation import *
import argparse
import datetime
import numpy as np
import warnings
from utils.focalloss import FocalLoss
from utils.Data_Load_with_Dail import *
from utils.model.BSVI_model import *
from utils.model.Resnet_row import resnet50
from utils.model.VIT import satellite_VIT
from utils.model.efficientNet_row import efficientnet_b0 as SV_efficientnet
from utils.model.VGG import vgg
from utils.model.densenet import densenet121
from utils.model.swin import swin_tiny_patch4_window7_224 as swin_model
from utils.model.R2Plus1D import R2Plus1D


warnings.filterwarnings("ignore")
# from utils.model.ResNet import *
from utils.random_seed import setup_seed
from utils.model.VIT import *
from utils.model.Fusion_model import *
import yaml

"""
使用的数据加载
1.分两种情况进行，只需在yaml进行修改
"""
def Load_Data(train_inf,val_inf,Data_root_path,train_batch_size=64,val_batch_size=64,num_threads=4):
    train_eii = DataSource(batch_size=train_batch_size, Datainf=train_inf, Data_root_path=Data_root_path)
    train_pipe = SourcePipeline(batch_size=train_batch_size, num_threads=num_threads, device_id=0, external_data=train_eii,
                                modeltype='train')
    train_iter = CustomDALIGenericIterator(len(train_eii) / train_batch_size, pipelines=[train_pipe],
                                           output_map=["satellite_img", 'Scene_sequence', 'Label'],
                                           last_batch_padded=False,
                                           size=len(train_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)
    val_eii = DataSource(batch_size=val_batch_size, Datainf=val_inf, Data_root_path=Data_root_path)
    val_pipe = SourcePipeline(batch_size=val_batch_size, num_threads=num_threads, device_id=0, external_data=val_eii,
                                modeltype='val')
    val_iter = CustomDALIGenericIterator(len(val_eii) / val_batch_size, pipelines=[val_pipe],
                                           output_map=["satellite_img", 'Scene_sequence', 'Label'],
                                           last_batch_padded=False,
                                           size=len(val_eii),
                                           last_batch_policy=LastBatchPolicy.PARTIAL,
                                           auto_reset=True)

    train_loader = train_iter
    val_loader = val_iter
    return train_loader,val_loader


if __name__ == '__main__':

    """
    数据文件加载
    """
    Data_root_path = 'Dataset/'
    train_inf = [json.loads(line) for line in open(Data_root_path + 'train.json')][0]
    val_inf = [json.loads(line) for line in open(Data_root_path + 'val.json')][0]
    """
    循环进行对config文件进行训练
    """
    config_path = 'configs/other_model.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    for cfg in config:
        """
        先定义模型
        """
        setup_seed(cfg['seed'])
        """
        这个是针对于卫星分支的消融实验
        """
        if cfg['model']['Branch'] == 'SFRAN':
            model = SFRAN(include_top=cfg['model']['include_top'],use_FAFN=cfg['model']['use_FAFN'],
                          use_LEU=cfg['model']['use_LEU'],use_GCT=cfg['model']['use_GCT']).cuda()
            experiment_name = 'SFRAN'
            for key, value in cfg['model'].items():
                if value is not False and value != 'SFRAN' and key != 'include_top' and key != 'Data_Branch':
                    experiment_name += key.replace('use','')
        """
        这个是针对卫星的其他模型
        """
        if cfg['model']['Data_Branch'] == 'satellite':
            if cfg['model']['Branch'] == 'resnet50':
                model = resnet50(num_classes=cfg['model']['num_classes']).cuda()
            if cfg['model']['Branch'] == 'satellite_VIT':
                model = satellite_VIT().cuda()
            if cfg['model']['Branch'] == 'efficientnet_b0':
                model = efficientnet_b0(num_classes=cfg['model']['num_classes']).cuda()
            if cfg['model']['Branch'] == 'vgg':
                model = vgg(num_classes=cfg['model']['num_classes']).cuda()
            if cfg['model']['Branch'] == 'swin':
                model = swin_model(num_classes=cfg['model']['num_classes']).cuda()
            if cfg['model']['Branch'] == 'densenet121':
                model = densenet121(num_classes=cfg['model']['num_classes']).cuda()
            experiment_name = 'satellite_'+cfg['model']['Branch']
        """
        这个是针对街景分支的消融实验
        """
        if cfg['model']['Branch'] == 'Trans_CFCCNN':
            model = Trans_CFCCNN(RNN_type=cfg['model']['RNN_type'],CA_FLAG=cfg['model']['CA_FLAG'],include_top = cfg['model']['include_top'] ).cuda()
            experiment_name = cfg['model']['Branch'] +'_'+ cfg['model']['RNN_type']
            if cfg['model']['RNN_type'] == 'ConvBiGRU' and cfg['model']['CA_FLAG'] == True:
                experiment_name = experiment_name + '_SA'
        """
        这个时针对街景分支的其他实验
        """
        if cfg['model']['Data_Branch'] == 'SV' and cfg['model']['Branch'] != 'Trans_CFCCNN':
            if cfg['model']['Branch'] == 'resnet50':
                model = resnet50(num_classes=cfg['model']['num_classes'],SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'satellite_VIT':
                model = SV_VIT(SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'efficientnet_b0':
                model = SV_efficientnet(num_classes=cfg['model']['num_classes'],SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'vgg':
                model = vgg(num_classes=cfg['model']['num_classes'],SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'swin':
                model = swin_model(num_classes=cfg['model']['num_classes'],SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'densenet121':
                model = densenet121(num_classes=cfg['model']['num_classes'],SV_Branch=cfg['model']['SV_Branch']).cuda()
            if cfg['model']['Branch'] == 'R2Plus1D':
                model = R2Plus1D(num_classes=cfg['model']['num_classes']).cuda()
            experiment_name = cfg['model']['Branch']
        """
        这个是针对整体模型的消融实验
        """
        if cfg['model']['Branch'] == 'MBFAF' and config_path.split('/')[1].replace('.yaml','') != 'hyperparam_ablation':
            model = TCFGC(MFSA_flag=cfg['model']['MFSA_flag'],MFOA_flag=cfg['model']['MFOA_flag'],HFMF_flag=cfg['model']['HFMF_flag']).cuda()
            experiment_name = cfg['model']['Branch']
            for key, value in cfg['model'].items():
                if value is not False and value != 'MBFAF'  and key != 'Data_Branch':
                    experiment_name += "_" + key.replace('flag','')
            if experiment_name == 'MBFAF':
                experiment_name = experiment_name + '_Cat'
        """
        这个是针对超参数的选择实验
        """
        if config_path.split('/')[1].replace('.yaml','') == 'hyperparam_ablation':
            model = TCFGC(MFSA_flag=cfg['model']['MFSA_flag'], MFOA_flag=cfg['model']['MFOA_flag'],HFMF_flag=cfg['model']['HFMF_flag']).cuda()
            experiment_name = cfg['model']['Branch'] +'_'+ str(cfg['learning_rate'])+'_'+str(cfg['weight_decay'])
            try:
                experiment_name = experiment_name +'_' +str(cfg['momentum'])
            except:
                pass
        """
        这个是针对整体模型的其他模型选择实验
        """
        if cfg['model']['Data_Branch'] == 'All':
            if cfg['model']['Branch'] == 'all_resnet50':
                model = resnet50(num_classes=cfg['model']['num_classes'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            if cfg['model']['Branch'] == 'all_VIT':
                model = SV_VIT(SV_Branch=cfg['model']['SV_Branch'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            if cfg['model']['Branch'] == 'all_efficientnet_b0':
                model = SV_efficientnet(num_classes=cfg['model']['num_classes'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            if cfg['model']['Branch'] == 'all_vgg':
                model = vgg(num_classes=cfg['model']['num_classes'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            if cfg['model']['Branch'] == 'all_swin':
                model = swin_model(num_classes=cfg['model']['num_classes'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            if cfg['model']['Branch'] == 'all_densenet121':
                model = densenet121(num_classes=cfg['model']['num_classes'],Data_Branch=cfg['model']['Data_Branch']).cuda()
            experiment_name = cfg['model']['Branch']
        """
        定义模型参数
        """
        if cfg['optimizer'] == 'adamw':
            try:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], betas=(cfg['momentum'],0.99),eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
            except:
                optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
        if cfg['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=float(cfg['eps']), weight_decay=float(cfg['weight_decay']))
        if cfg['scheduler_flag']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg['train_epoch'], eta_min=cfg['eta_min'])
        loss_function = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']).cuda()
        best_accident_acc = 0.0
        model_weight_path = os.path.join('model',experiment_name+'.pth')
        """
        数据加载
        """
        train_loader, val_loader = Load_Data(train_inf=train_inf, val_inf=val_inf, Data_root_path=Data_root_path,train_batch_size=cfg['train_batch_size'],
                                             val_batch_size=cfg['val_batch_size'], num_threads=12)
        train_numbers, val_numbers = len(train_inf), len(val_inf)
        """
        训练
        """
        log_time = datetime.datetime.now().strftime('%d_%H_%M')
        writer = SummaryWriter("exp/" + config_path.split('/')[1].replace('.yaml','') +'/'+ experiment_name, flush_secs=60)
        for epoch in range(cfg['train_epoch']):
            sum_train_loss = 0.0
            train_accident_acc = 0.0
            train_overspeed_acc = 0.0
            train_loss_dict = {'accident_loss': [], 'overspeed_loss': []}
            model.train()
            train_bar = tqdm(train_loader, file=sys.stdout, ncols=200, position=0)
            for step, batch in enumerate(train_bar):
                """
                每次开始前将梯度清零
                """
                optimizer.zero_grad()
                """
                获得训练的结果和loss，进行反向传播并更新梯度
                """
                Scene_img, Scene_sequence, overspeed_label = batch['satellite_img'], batch['Scene_sequence'], batch['Label']
                if cfg['model']['Data_Branch'] == 'satellite':
                    output_accident = model(Scene_img)
                if cfg['model']['Data_Branch'] == 'SV':
                    output_accident = model(Scene_sequence)
                if cfg['model']['Data_Branch'] == 'All':
                    output_accident = model(Scene_sequence,Scene_img)
                loss = loss_function(output_accident, overspeed_label)
                loss.backward()
                optimizer.step()
                """
                梯度清零
                """
                optimizer.zero_grad()
                """
                记录迭代后的loss值
                loss可能需要打印一次
                """
                sum_train_loss = sum_train_loss + loss.item()
                """
                计算训练期间的验证指标值
                """
                train_predict_accident = torch.max(output_accident, dim=1)[1]
                train_accident_acc += torch.eq(train_predict_accident, overspeed_label).sum().item()
                """
                打印当前迭代下的loss、验证指标值
                """
                train_bar.desc = '训练阶段==> Loss:{:.3f}'.format(loss.item())
            if cfg['scheduler_flag']:
                scheduler.step()
            epoch_times = step + 1
            print(experiment_name+' [Train epoch %d] 训练阶段平均指标======>Train_SUM_Loss: %.3f  事故分类精度: %.3f ' % (
            epoch + 1, sum_train_loss / epoch_times, train_accident_acc / train_numbers))
            writer.add_scalars('训练指标', {"Loss": round(sum_train_loss / epoch_times, 3),"Accident_acc": round(train_accident_acc / train_numbers, 3)}, epoch + 1)
            if (epoch + 1) % cfg['train_val_times'] == 0:
                TP, TN, FP, FN = 0, 0, 0, 0
                model.eval()
                sum_val_loss = 0.0
                val_accident_acc = 0.0
                with torch.no_grad():
                    val_bar = tqdm(val_loader, file=sys.stdout, ncols=200, position=0)
                    for step, batch in enumerate(val_bar):
                        optimizer.zero_grad()
                        Scene_img, Scene_sequence, overspeed_label = batch['satellite_img'], batch['Scene_sequence'],batch['Label']
                        if cfg['model']['Data_Branch'] == 'satellite':
                            output_accident = model(Scene_img)
                        if cfg['model']['Data_Branch'] == 'SV':
                            output_accident = model(Scene_sequence)
                        if cfg['model']['Data_Branch'] == 'All':
                            output_accident = model(Scene_sequence, Scene_img)
                        loss = loss_function(output_accident, overspeed_label)
                        sum_val_loss = sum_val_loss + loss.item()
                        val_predict_accident = torch.max(output_accident, dim=1)[1]
                        predict_cla, target = val_predict_accident.item(), overspeed_label.item()
                        val_accident_acc += torch.eq(val_predict_accident, overspeed_label).sum().item()
                        if predict_cla == 0 and target == 0:
                            TP += 1
                        if predict_cla == 1 and target == 1:
                            TN += 1
                        if predict_cla == 0 and target == 1:
                            FP += 1
                        if predict_cla == 1 and target == 0:
                            FN += 1
                        val_bar.desc = '验证阶段==> Loss:{:.3f}'.format(loss.item())
                    try:
                        P = TP / (TP + FP)
                        R = TP / (TP + FN)
                        F1 = 2 * P * R / (P + R)
                        acc = (TP + TN) / (TP + TN + FP + FN)
                    except:
                        P, R, F1, acc = 0, 0, 0, 0
                    epoch_times = step + 1
                    print('[Train epoch %d] 验证阶段平均指标======>Val_SUM_Loss: %.3f  超速分类精度: %.3f ' % (
                    epoch + 1, sum_val_loss / epoch_times, val_accident_acc / val_numbers))
                    if (best_accident_acc <= val_accident_acc / val_numbers):
                        best_accident_acc = val_accident_acc / val_numbers
                        torch.save(model.state_dict(), model_weight_path)
                    writer.add_scalars('验证指标', {"Loss": round(sum_val_loss / epoch_times, 3),
                                                "Accident_acc": round(val_accident_acc / val_numbers, 3),
                                                "P": round(P, 3),
                                                "R": round(R, 3),
                                                "F1": round(F1, 3), }, epoch + 1)