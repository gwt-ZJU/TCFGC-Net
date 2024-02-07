import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from einops import rearrange
from SATL import *
from pool import *
from efficientNet import *
from convLSTM import *
from einops.layers.torch import Rearrange, Reduce
from GRU import *

class satellite_model(nn.Module):
    def __init__(self):
        super(satellite_model, self).__init__()
        layers = [2, 2, 2, 2]

        self.satellite_model = CFTResNet(CFT_Bottleneck,layers=layers,num_classes=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1 = nn.Conv2d(in_channels=2048,out_channels=1024,kernel_size=1,padding=0,stride=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=2, bias=True)
        )

    def forward(self,x):
        out = self.satellite_model(x)
        # out = self.conv_1x1(out)
        # out = self.avgpool(out)
        return out

class BSVI_model(nn.Module):
    def __init__(self):
        super(BSVI_model, self).__init__()
        """
        一些配置参数设置
        """
        width_coefficient = 1.0
        depth_coefficient = 1.1
        dropout_rate = 0.2
        num_classes = 2
        """
        街景图像的特征提取架构
        """
        self.view_1 = EfficientNet_BSVI(width_coefficient = width_coefficient,depth_coefficient=depth_coefficient,
                                          dropout_rate=dropout_rate,num_classes=num_classes)
        self.view_2 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)
        self.view_3 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)
        self.view_4 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)
        """
        BAM-BiConvLstm架构
        """

        self.bam_bilstm = ConvBLSTM(in_channels=192, hidden_channels=384, kernel_size=(3, 3), num_layers=1, batch_first=True)

        self.conv1 = nn.Conv2d(in_channels=384*4,out_channels=512,kernel_size=1)

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.conv3d = nn.Conv3d(in_channels=384, out_channels=512, kernel_size=(4, 1, 1),
                                stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=2, bias=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):
        bsvi_1 = x[:,0,:,:,:]
        bsvi_2 = x[:, 1, :, :, :]
        bsvi_3 = x[:, 2, :, :, :]
        bsvi_4 = x[:, 3, :, :, :]
        """
        四个不同分支的特征抽取
        """
        out1 = self.view_1(bsvi_1)
        out2 = self.view_2(bsvi_2)
        out3 = self.view_3(bsvi_3)
        out4 = self.view_4(bsvi_4)
        """
        进行BAM-BiConvLSTM进行处理
        """
        x_fwd = torch.stack([out1, out2, out3, out4], dim=1)
        x_rev = torch.stack([out4, out3, out2, out1], dim=1)
        out = self.bam_bilstm(x_fwd,x_rev)
        # out = x_fwd
        """
        lstm的输出结果进行，cat然后变成1D数据
        1.cat和维度变化
        2.变成1D数据进行后续操作
        """
        #1
        b,t,c,h,w = out.shape
        out = rearrange(out,'b t c h w -> b (t c) h w')
        # out = rearrange(out, 'b t c h w -> b c t h w')
        # out = self.conv3d(out)
        # out = out.squeeze(dim=2)

        out = self.conv1(out)
        #2
        # out = self.avg(out)
        # out = torch.flatten(out,1)
        # out = self.fc(out)
        return out

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x + x
        return out

class MFSA(nn.Module):
    def __init__(self):
        super(MFSA, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x + x
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MFOA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MFOA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x) + x
        return out

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            #由此可以看出 FeedForward 的输入和输出维度是一致的
            nn.Linear(dim,hidden_dim),
            #激活函数
            nn.GELU(),
            #防止过拟合
            nn.Dropout(dropout),
            #重复上述过程
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')

        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x

class Mixer(nn.Module):
    def __init__(self,feature_size = 8,patch_size=1,inchannel=512,depth=1,token_dim=512,dropout=0.1):
        super(Mixer, self).__init__()
        self.num_patches = (feature_size // patch_size) ** 2
        self.to_emb = Rearrange('b c h w -> b (h w) c')
        self.to_embedding = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
            )
        self.layer_normal = nn.LayerNorm(inchannel)
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim=inchannel, num_patch=self.num_patches, token_dim=token_dim,
                                                channel_dim=1024, dropout=dropout))
        self.size_reshape = nn.Sequential(
            Rearrange('b (h w) c -> b c h w',h=feature_size // patch_size,w=feature_size // patch_size),
            # nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=patch_size, stride=patch_size),

            )

    def forward(self,x):
        _,_,H,W =x.shape #512*8*8
        x = self.to_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)
        x = self.size_reshape(x)
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


class Fusion_Module(nn.Module):
    def __init__(self):
        super(Fusion_Module, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1),
            nn.GELU(),
            nn.BatchNorm2d(512))

        self.fusion_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=512*3, out_channels=512, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(512))

        self.satellite_conv = nn.Conv1d(in_channels=64,out_channels=48,kernel_size=1)
        self.bsvi_up = nn.Upsample(size=(8,8))

        # self.bsvi_Spatial = SpatialAttentionModule()
        # self.satellite_Spatial = SpatialAttentionModule()

        self.bsvi_Spatial = SELayer(channel=512)
        self.satellite_Spatial = SELayer(channel=512)
        self.mixer_mlp = Mixer(feature_size=8,patch_size=1,inchannel=512,depth=1,token_dim=512,dropout=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,satellite_feature,bsvi_feature):
        # bsvi_feature = self.bsvi_up(bsvi_feature)

        """
        各自通过空间注意力
        """
        bsvi_feature_Spatial = self.bsvi_Spatial(bsvi_feature)
        satellite_feature_Spatial = self.satellite_Spatial(satellite_feature)
        """
        cat后进行卫星和bsvi信息学习,用的是mlp，就是转成一维特征
        """
        fusion_feature = torch.cat([satellite_feature, bsvi_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)  #这个地方是 1024,8,8
        branch_fusion = fusion_feature
        fusion_feature = self.mixer_mlp(fusion_feature) + branch_fusion
        """
        融合特征和分支融合
        """
        fusion_feature = torch.cat([bsvi_feature_Spatial,satellite_feature_Spatial,fusion_feature],dim=1)
        fusion_feature = self.fusion_conv_2(fusion_feature)
        # b,c,h,w = fusion_feature.shape
        # bsvi_fusion_feature = fusion_feature[:,:int(c/2),:,:]
        # satellite_fusion_feature = fusion_feature[:, int(c/2):, :, :]
        #
        # bsvi_fusion_feature = bsvi_feature_Spatial * fusion_feature
        # satellite_fusion_feature = satellite_feature_Spatial * fusion_feature
        # bsvi_fusion_feature = bsvi_feature_Spatial
        # satellite_fusion_feature = satellite_feature_Spatial
        """
        cat
        """
        # fusion_feature_2 = torch.cat([satellite_fusion_feature, bsvi_fusion_feature], dim=1)

        # fusion_feature_2 = rearrange(fusion_feature_2,'b c (h w) -> b c h w')

        return fusion_feature


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.satellite_encoder = satellite_model()
        self.bsvi_model = BSVI_model()
        self.fusion_module = Fusion_Module()
        self.avg_pool_1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2, bias=True),
        )

    def forward(self,bsvi,satellite):
        satellite_feature = self.satellite_encoder(satellite)
        bsvi_feature = self.bsvi_model(bsvi)

        out = self.fusion_module(satellite_feature,bsvi_feature)
        out = self.avg_pool_2(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out

"""
这里定义下我的模型 双分支上下文特征引导融合网络
"""
class TCFGC(nn.Module):
    def __init__(self,MFSA_flag = True,MFOA_flag = True,HFMF_flag = True):
        super(TCFGC, self).__init__()
        self.satellite_encoder = SFRAN(include_top=False,use_FAFN=True,use_GCT=True,use_LEU=True)
        self.bsvi_model = Trans_CFCCNN(include_top=False,RNN_type='ConvBiGRU',CA_FLAG=True)
        self.fusion_module = MBFAF(MFSA_flag = MFSA_flag,MFOA_flag = MFOA_flag,HFMF_flag = HFMF_flag)
        self.avg_pool_1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2, bias=True),
        )

    def forward(self,bsvi,satellite):
        satellite_feature = self.satellite_encoder(satellite)
        bsvi_feature = self.bsvi_model(bsvi)

        out = self.fusion_module(satellite_feature,bsvi_feature)
        out = self.avg_pool_2(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out

class Trans_CFCCNN(nn.Module):
    def __init__(self,RNN_type='ConvBiGRU',CA_FLAG=True,include_top=False):
        super(Trans_CFCCNN, self).__init__()
        """
        一些配置参数设置
        """
        width_coefficient = 1.0
        depth_coefficient = 1.1
        dropout_rate = 0.2
        num_classes = 2
        self.RNN_type = RNN_type
        self.CA_FLAG = CA_FLAG
        """
        街景图像的特征提取架构
        """
        self.view_1 = EfficientNet_BSVI(width_coefficient = width_coefficient,depth_coefficient=depth_coefficient,
                                          dropout_rate=dropout_rate,num_classes=num_classes)
        self.view_2 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)
        self.view_3 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)
        self.view_4 = EfficientNet_BSVI(width_coefficient=width_coefficient, depth_coefficient=depth_coefficient,
                                        dropout_rate=dropout_rate, num_classes=num_classes)


        if self.RNN_type == 'ConvLSTM':
            self.ConvLSTM = ConvLSTM(in_channels=192, hidden_channels=[192,192], kernel_size=(3, 3), num_layers=2,batch_first=True, RNN_type=self.RNN_type)
            self.conv1 = nn.Conv2d(in_channels=192*4,out_channels=512,kernel_size=1)
        if self.RNN_type == 'ConvBiLSTM':
            self.ConvLSTM = ConvLSTM(in_channels=192, hidden_channels=[192,192], kernel_size=(3, 3), num_layers=2,batch_first=True, RNN_type=self.RNN_type)
            self.conv1 = nn.Conv2d(in_channels=384*4,out_channels=512,kernel_size=1)
        if self.RNN_type == 'ConvGRU':
            self.ConvGRU = ConvGRU(in_channels=192,hidden_channels=[192,192],kernel_size=(3, 3),num_layers=2,batch_first=True,RNN_type=self.RNN_type,CA_FLAG=CA_FLAG)
            self.conv1 = nn.Conv2d(in_channels=192 * 4, out_channels=512, kernel_size=1)
        if self.RNN_type == 'ConvBiGRU':
            self.ConvGRU = ConvGRU(in_channels=192,hidden_channels=[192,192],kernel_size=(3, 3),num_layers=2,batch_first=True,RNN_type=self.RNN_type,CA_FLAG=CA_FLAG)
            self.conv1 = nn.Conv2d(in_channels=384 * 4, out_channels=512, kernel_size=1)
        if self.RNN_type == 'None':
            self.conv1 = nn.Conv2d(in_channels=192 * 4, out_channels=512, kernel_size=1)
        if self.RNN_type == 'LSTM':
            self.conv1 = nn.Linear(in_features=192*8*8, out_features=64*8*8)
            self.LSTM = nn.LSTM(input_size=64*8*8,hidden_size=64*8*8, num_layers=2,bidirectional=False,batch_first=True)
            self.conv2 = nn.Conv2d(in_channels=4*64, out_channels=512, kernel_size=1)
        if self.RNN_type == 'GRU':
            self.conv1 = nn.Linear(in_features=192 * 8 * 8, out_features=64 * 8 * 8)
            self.GRU = nn.GRU(input_size=64 * 8 * 8, hidden_size=64 * 8 * 8, num_layers=2, bidirectional=False,batch_first=True)
            self.conv2 = nn.Conv2d(in_channels=4 * 64, out_channels=512, kernel_size=1)


        self.avg = nn.AdaptiveAvgPool2d(1)

        self.conv3d = nn.Conv3d(in_channels=384, out_channels=512, kernel_size=(4, 1, 1),
                                stride=(1, 1, 1), padding=(0, 0, 0), bias=False)

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=2, bias=True)
        )

        self.include_top = include_top

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):
        bsvi_1 = x[:,0,:,:,:]
        bsvi_2 = x[:, 1, :, :, :]
        bsvi_3 = x[:, 2, :, :, :]
        bsvi_4 = x[:, 3, :, :, :]
        """
        四个不同分支的特征抽取
        """
        out1 = self.view_1(bsvi_1)
        out2 = self.view_2(bsvi_2)
        out3 = self.view_3(bsvi_3)
        out4 = self.view_4(bsvi_4)
        """
        这部分是进行RNN结构的处理
        """
        x_fwd = torch.stack([out1, out2, out3, out4], dim=1)
        x_rev = torch.stack([out4, out3, out2, out1], dim=1)
        if self.RNN_type == 'ConvLSTM':
            out = self.ConvLSTM(x_fwd, x_rev)
            out = rearrange(out,'b t c h w -> b (t c) h w')
            out = self.conv1(out)
        if self.RNN_type == 'ConvBiLSTM':
            out = self.ConvLSTM(x_fwd, x_rev)
            out = rearrange(out, 'b t c h w -> b (t c) h w')
            out = self.conv1(out)
        if self.RNN_type == 'ConvGRU':
            out = self.ConvGRU(x_fwd, x_rev)
            out = rearrange(out, 'b t c h w -> b (t c) h w')
            out = self.conv1(out)
        if self.RNN_type == 'ConvBiGRU':
            out = self.ConvGRU(x_fwd, x_rev)
            out = rearrange(out, 'b t c h w -> b (t c) h w')
            out = self.conv1(out)
        if self.RNN_type == 'None':
            out = rearrange(x_fwd, 'b t c h w -> b (t c) h w')
            out = self.conv1(out)
        if self.RNN_type == 'LSTM':
            b, t, c ,h ,w = x_fwd.shape
            x_fwd = rearrange(x_fwd, 'b t c h w -> b t (c h w)')
            out = self.conv1(x_fwd)
            out, (hn, cn) = self.LSTM(out)
            out = rearrange(out, 'b t (c h w) -> b (t c) h w',c=64,h=h,w=w)
            out = self.conv2(out)
        if self.RNN_type == 'GRU':
            b, t, c, h, w = x_fwd.shape
            x_fwd = rearrange(x_fwd, 'b t c h w -> b t (c h w)')
            out = self.conv1(x_fwd)
            out, hn = self.GRU(out)
            out = rearrange(out, 'b t (c h w) -> b (t c) h w', c=64, h=h, w=w)
            out = self.conv2(out)
        """
        是否单分支输出
        """
        if self.include_top:
            out = self.avg(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
        else:
            return out

class SFRAN(nn.Module):
    def __init__(self,include_top=False,use_LEU=True,use_GCT=True,use_FAFN=True,embed_dims=[32, 64, 128], num_heads=[1, 4, 8], mlp_ratios=[4, 4, 4],
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2], sr_ratios=[8, 4, 2],
                 drop_rate=0.0, drop_path_rate=0.1):
        super(SFRAN, self).__init__()
        layers = [2, 2, 2, 2]
        self.satellite_model = CFTResNet(CFT_Bottleneck,layers=layers,num_classes=2,use_LEU=use_LEU,use_GCT=use_GCT,use_FAFN=use_FAFN,embed_dims=embed_dims, num_heads=num_heads,
                                         mlp_ratios=mlp_ratios,qkv_bias=qkv_bias, norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2, bias=True)
        )
        self.include_top = include_top

    def forward(self,x):
        out = self.satellite_model(x)
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out
        else:
            return out

class MBFAF(nn.Module):
    def __init__(self,MFSA_flag = True,MFOA_flag = True,HFMF_flag = True):
        super(MBFAF, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=1,stride=1),
            nn.GELU(),
            nn.BatchNorm2d(512))

        self.fusion_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=512*3, out_channels=512, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(512))

        self.satellite_conv = nn.Conv1d(in_channels=64,out_channels=48,kernel_size=1)
        self.bsvi_up = nn.Upsample(size=(8,8))

        # self.bsvi_Spatial = SpatialAttentionModule()
        # self.satellite_Spatial = SpatialAttentionModule()

        self.MFSA_flag = MFSA_flag
        self.MFOA_flag = MFOA_flag
        self.HFMF_flag = HFMF_flag
        if self.MFOA_flag:
            self.MFOA = MFOA(channel=512)
        if self.MFSA_flag:
            self.MFSA = MFSA()
        if self.HFMF_flag:
            self.mixer_mlp = Mixer(feature_size=8,patch_size=1,inchannel=512,depth=1,token_dim=512,dropout=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,satellite_feature,bsvi_feature):
        # bsvi_feature = self.bsvi_up(bsvi_feature)

        """
        各自通过空间注意力
        """
        if self.MFOA_flag:
            bsvi_feature_Spatial = self.MFOA(bsvi_feature)
        if self.MFOA_flag is not True:
            bsvi_feature_Spatial = bsvi_feature

        if self.MFSA_flag:
            satellite_feature_Spatial = self.MFSA(satellite_feature)
        if self.MFSA_flag is not True:
            satellite_feature_Spatial = satellite_feature
        """
        cat后进行卫星和bsvi信息学习,用的是mlp，就是转成一维特征
        """
        if self.HFMF_flag:
            fusion_feature = torch.cat([satellite_feature, bsvi_feature], dim=1)
            fusion_feature = self.fusion_conv(fusion_feature)  #这个地方是 1024,8,8
            branch_fusion = fusion_feature
            fusion_feature = self.mixer_mlp(fusion_feature) + branch_fusion
        if self.HFMF_flag is not True:
            fusion_feature = torch.cat([satellite_feature, bsvi_feature], dim=1)
            fusion_feature = self.fusion_conv(fusion_feature)
        """
        融合特征和分支融合
        """
        fusion_feature = torch.cat([bsvi_feature_Spatial,satellite_feature_Spatial,fusion_feature],dim=1)
        fusion_feature = self.fusion_conv_2(fusion_feature)
        # b,c,h,w = fusion_feature.shape
        # bsvi_fusion_feature = fusion_feature[:,:int(c/2),:,:]
        # satellite_fusion_feature = fusion_feature[:, int(c/2):, :, :]
        #
        # bsvi_fusion_feature = bsvi_feature_Spatial * fusion_feature
        # satellite_fusion_feature = satellite_feature_Spatial * fusion_feature
        # bsvi_fusion_feature = bsvi_feature_Spatial
        # satellite_fusion_feature = satellite_feature_Spatial
        """
        cat
        """
        # fusion_feature_2 = torch.cat([satellite_fusion_feature, bsvi_fusion_feature], dim=1)

        # fusion_feature_2 = rearrange(fusion_feature_2,'b c (h w) -> b c h w')

        return fusion_feature

# if __name__ == '__main__':
#     """
#     用ResNet50的block数量结构
#     """
#     bsvi = torch.rand(2,4,3,256,256).cuda()
#     satellite = torch.rand(2,3,256,256).cuda()
#     """
#     现在研究卫星分支
#     """
#     # model = SFRAN(include_top=True,use_GCT=True,use_FAFN=True,use_LEU=False,embed_dims=[32, 64, 128], num_heads=[1, 4, 8], mlp_ratios=[4, 4, 4],
#     #              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2], sr_ratios=[8, 4, 2],
#     #              drop_rate=0.0, drop_path_rate=0.1).cuda()
#     # model = SFRAN(include_top=True, use_GCT=False, use_FAFN=True, use_LEU=True).cuda()
#     # model = Trans_CFCCNN(RNN_type='GRU',CA_FLAG=False,include_top=True).cuda()
#     # out = model(bsvi,satellite)
#     model = TCFGC(MFSA_flag=False,MFOA_flag=False,HFMF_flag=False).cuda()
#     out = model(bsvi,satellite)
#     pass
