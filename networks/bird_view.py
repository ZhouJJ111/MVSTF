import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone            # modify
# import backbone            # modify
import pdb
from icecream import ic

class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.merge_layer = nn.Sequential(
                    backbone.conv3x3(cin, cin // 2, stride=1, dilation=1),
                    nn.BatchNorm2d(cin // 2),
                    backbone.act_layer,
                    
                    backbone.conv3x3(cin // 2, cout, stride=1, dilation=1),
                    nn.BatchNorm2d(cout),
                    backbone.act_layer
                )
        self.scale_factor = scale_factor
    
    def forward(self, x_low, x_high):
        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)
        x_out = self.merge_layer(x_merge)
        return x_out


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.att_layer = nn.Sequential(
            backbone.conv3x3(2 * cout, cout // 2, stride=1, dilation=1),
            nn.BatchNorm2d(cout // 2),
            backbone.act_layer,
            backbone.conv3x3(cout // 2, 2, stride=1, dilation=1, bias=True)
        )

        self.conv_high = nn.Sequential(
            backbone.conv3x3(cin_high, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )

        self.conv_low = nn.Sequential(
            backbone.conv3x3(cin_low, cout, stride=1, dilation=1),
            nn.BatchNorm2d(cout),
            backbone.act_layer
        )
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        batch_size = x_low.shape[0]
        H = x_low.shape[2]
        W = x_low.shape[3]

        x_high_up = F.upsample(x_high, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        x_merge = torch.stack((self.conv_low(x_low), self.conv_high(x_high_up)), dim=1) #(BS, 2, channels, H, W)
        x_merge = F.dropout(x_merge, p=0.2, training=self.training, inplace=False)

        # attention fusion
        ca_map = self.att_layer(x_merge.view(batch_size, 2*self.cout, H, W))
        ca_map = ca_map.view(batch_size, 2, 1, H, W)
        ca_map = F.softmax(ca_map, dim=1)

        x_out = (x_merge * ca_map).sum(dim=1) #(BS, channels, H, W)
        return x_out


# ASPP: https://github.com/fregu856/deeplabv3/blob/master/model/aspp.py
class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.middle_channel = in_channel // 2

        # conv 1*1
        self.conv_1x1_1 = nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(self.middle_channel)

        # 3x3 convolutions with different dilation rates
        self.conv_3x3_1 = nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(self.middle_channel)

        self.conv_3x3_2 = nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(self.middle_channel)

        self.conv_3x3_3 = nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=3, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(self.middle_channel)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(self.middle_channel)

        # Final 1x1 convolution to combine all the features
        self.conv_1x1_3 = nn.Conv2d(self.middle_channel * 5, self.middle_channel, kernel_size=1)  # 5 branches
        self.bn_conv_1x1_3 = nn.BatchNorm2d(self.middle_channel)

        self.conv_1x1_4 = nn.Conv2d(self.middle_channel, self.out_channel, kernel_size=1)



    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))                      # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))
        out = self.conv_1x1_4(out) # (shape: (batch_size, num_classes, h/16, w/16))

        return out



# MSCA: https://github.com/fengluodb/LENet
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv3_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv4 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv1_1(attn)
        attn_0 = self.conv1_2(attn_0)

        attn_1 = self.conv2_1(attn)
        attn_1 = self.conv2_2(attn_1)

        attn_2 = self.conv3_1(attn)
        attn_2 = self.conv3_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv4(attn)

        return attn * x


# MAFL: https://github.com/CVMI-Lab/MarS3D
class MAFL(nn.Module):
    def __init__(self, in_channel,  middle_channel, out_channel):
        super(MAFL,self).__init__()
        self.in_channel = in_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel

        self.p0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=1),                     # 1*1
            nn.ReLU()
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, kernel_size=1),  # 1*1
            nn.ReLU()
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 3), padding=(0, 1), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (3, 1), padding=(1, 0), groups=self.middle_channel),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 5), padding=(0, 2), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (5, 1), padding=(2, 0), groups=self.middle_channel),
            nn.ReLU()
        )

        self.p4 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 7), padding=(0, 3), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (7, 1), padding=(3, 0), groups=self.middle_channel),
            nn.ReLU()
        )

        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),                               # 最大池化
            nn.Conv2d(self.middle_channel,  self.middle_channel, kernel_size=1),
            nn.ReLU()
        )

        self.conv4 = nn.Conv2d(5 * self.middle_channel, self.out_channel, 1)

    def forward(self, x):               # Input:[BS, 64*9, 512, 512],  # Output: [BS, 64*9, 512, 512]
        x_pre = self.p0(x)
        x1 = self.p1(x_pre)         # 1*1
        x2 = self.p2(x_pre)         # 3*3
        x3 = self.p3(x_pre)         # 5*5
        x4 = self.p4(x_pre)         # 7*7
        x5 = self.p5(x_pre)         # pool
        p = torch.cat((x1, x2, x3, x4, x5), dim=1)

        p = self.conv4(p)
        attn = p * x

        return attn

# MAFL: https://github.com/CVMI-Lab/MarS3D
class MAFL_BN(nn.Module):
    def __init__(self, in_channel,  middle_channel, out_channel):
        super(MAFL_BN,self).__init__()
        self.in_channel = in_channel
        self.middle_channel = middle_channel
        self.out_channel = out_channel

        self.p0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.middle_channel, kernel_size=1),                     # 1*1
            nn.BatchNorm2d(self.middle_channel),
            nn.ReLU()
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, kernel_size=1),  # 1*1
            nn.ReLU()
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 3), padding=(0, 1), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (3, 1), padding=(1, 0), groups=self.middle_channel),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 5), padding=(0, 2), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (5, 1), padding=(2, 0), groups=self.middle_channel),
            nn.ReLU()
        )

        self.p4 = nn.Sequential(
            nn.Conv2d(self.middle_channel, self.middle_channel, (1, 7), padding=(0, 3), groups=self.middle_channel),
            nn.Conv2d(self.middle_channel, self.middle_channel, (7, 1), padding=(3, 0), groups=self.middle_channel),
            nn.ReLU()
        )

        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),                               # 最大池化
            nn.Conv2d(self.middle_channel,  self.middle_channel, kernel_size=1),
            nn.ReLU()
        )

        # self.conv4 = nn.Conv2d(5 * self.middle_channel, self.out_channel, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(5 * self.middle_channel, self.out_channel, 1),
            nn.BatchNorm2d(self.out_channel),
        )



    def forward(self, x):               # Input:[BS, 64*9, 512, 512],  # Output: [BS, 64*9, 512, 512]
        x_pre = self.p0(x)
        x1 = self.p1(x_pre)         # 1*1
        x2 = self.p2(x_pre)         # 3*3
        x3 = self.p3(x_pre)         # 5*5
        x4 = self.p4(x_pre)         # 7*7
        x5 = self.p5(x_pre)         # pool
        p = torch.cat((x1, x2, x3, x4, x5), dim=1)

        p = self.conv4(p)
        attn = p * x

        return attn


class BEVNet(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(BEVNet, self).__init__()


        # encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1],
                                       layers[0], stride=2, dilation=1, use_att=use_att)
        # MAFL
        # self.header2 = MAFL(in_channel=context_layers[1], middle_channel=context_layers[2], out_channel=context_layers[1])
        self.header2 = MAFL_BN(in_channel=context_layers[1], middle_channel=context_layers[2],
                            out_channel=context_layers[1])
 
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1], context_layers[2],
                                     layers[1], stride=2, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2], context_layers[3],
                                     layers[2], stride=2, dilation=1, use_att=use_att)

        # decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)

        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)

        self.out_channels = fusion_channels1 // 2

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))

        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))

        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        # pdb.set_trace()
        # encoder
                                                # [1, 576, 512, 512]
        # MAFL

        x0 = self.header(x)                     # [1, 32, 256, 256]
        x02 = self.header2(x0)                   # [1, 64, 256, 256], modify: MAFL
        x1 = self.res1(x02)                      # [1, 64, 128, 128]
        x2 = self.res2(x1)                      # [1, 128, 64, 64]
        ic('BEVNet_Input:', x.shape,  x1.shape, x2.shape)
        ic('BEVNet_MAFL:', x0.shape, x02.shape)

        # decoder
        x_merge1 = self.up2(x1, x2)                             # [1, 96, 128, 128]
        x_merge = self.up1(x0, x_merge1)                        # [1, 64, 256, 256]
        ic('BEVNet_Output:',  x_merge1.shape, x_merge.shape)
        return x_merge



# 原始版本： Spatial attention + IAC
class BEVNet_new(nn.Module):
    def __init__(self, base_block, context_layers, layers, use_att):
        super(BEVNet_new, self).__init__()
        #encoder
        self.header = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[0], context_layers[1], layers[0], stride=2, dilation=1, use_att=use_att)
        self.res1 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[1], context_layers[2], layers[1], stride=2, dilation=1, use_att=use_att)
        self.res2 = self._make_layer(eval('backbone.{}'.format(base_block)), context_layers[2], context_layers[3], layers[2], stride=2, dilation=1, use_att=use_att)

        #decoder
        fusion_channels2 = context_layers[3] + context_layers[2]
        self.up2 = AttMerge(context_layers[2], context_layers[3], fusion_channels2 // 2, scale_factor=2)
        
        fusion_channels1 = fusion_channels2 // 2 + context_layers[1]
        self.up1 = AttMerge(context_layers[1], fusion_channels2 // 2, fusion_channels1 // 2, scale_factor=2)
        
        self.out_channels = fusion_channels1 // 2
    
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1, use_att=True):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        
        for i in range(num_blocks):
            layer.append(block(out_planes, dilation=dilation, use_att=False))
        
        layer.append(block(out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        #pdb.set_trace()
        #encoder                            #  x:[1,64,512,512]
        x0 = self.header(x)                 # x0:[1,32,256,256], 1/2
        x1 = self.res1(x0)                  # x1:[1,64,128,128], 1/2
        x2 = self.res2(x1)                  # x2:[1,128,64,64],  1/2
        
        #decoder
        x_merge1 = self.up2(x1, x2)         # x_merge1:[1,96,128,128], *2
        x_merge = self.up1(x0, x_merge1)    # x_merge: [1,64,256,256], *2
        return x_merge

if __name__=="__main__":
    # 假设输入特征图的形状为 (batch_size, 128, height, width)
    input_feature_map = torch.randn(2, 64, 512, 512)  # 假设的输入特征图
    # input_feature_map = torch.randn(2, 128, 64, 64)  # 假设的输入特征图



    # 创建 ASPP 模块实例
    aspp_module = ASPP(in_channel=64, out_channel=64)
    output_aspp = aspp_module(input_feature_map)
    print("ASPP output shape:", output_aspp.shape)

    # 创建 AttentionModule 模块实例
    attention_module = AttentionModule(dim=64)  # 假设输入特征图的通道数为 512
    output_attention = attention_module(input_feature_map)
    print("Attention output shape:", output_attention.shape)

    # 创建 MAFL 模块实例
    maf_module = MAFL(in_channel=64, middle_channel=128, out_channel=64)
    output_maf = maf_module(input_feature_map)
    print("MAFL output shape:", output_maf.shape)
