import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import deep_point

from utils.criterion import CE_OHEM
from utils.lovasz_losses import lovasz_softmax

import yaml
import copy
import pdb
from icecream import ic

def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)       # [512,512,30]
        self.rv_shape = list(pModel.Voxel.rv_shape)         # [64,2048]
        self.bev_wl_shape = self.bev_shape[:2]              # [512,512]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / (pModel.Voxel.bev_shape[0])     # [50-(-50)]/512
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / (pModel.Voxel.bev_shape[1])     # [50-(-50)]/512
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / (pModel.Voxel.bev_shape[2])     # [2-(-4)]/30

        self.point_feat_out_channels = pModel.point_feat_out_channels                                   # 64

        # modify
        self.seq_num = pModel.seq_num                                           # 9

        self.build_network()
        self.build_loss()

    def build_loss(self):
        self.criterion_seg_cate = None
        print("Loss mode: {}".format(self.pModel.loss_mode))
        if self.pModel.loss_mode == 'ce':
            self.criterion_seg_cate = nn.CrossEntropyLoss(ignore_index=0)
        elif self.pModel.loss_mode == 'ohem':
            self.criterion_seg_cate = CE_OHEM(top_ratio=0.2, top_weight=4.0, ignore_index=0)
        elif self.pModel.loss_mode == 'wce':
            content = torch.zeros(self.pModel.class_num, dtype=torch.float32)
            with open('datasets/semantic-kitti-mos.yaml', 'r') as f:
                task_cfg = yaml.load(f)
                for cl, freq in task_cfg["content"].items():
                    x_cl = task_cfg['learning_map'][cl]
                    content[x_cl] += freq

            loss_w = 1 / (content + 0.001)
            loss_w[0] = 0
            print("Loss weights from content: ", loss_w)
            self.criterion_seg_cate = nn.CrossEntropyLoss(weight=loss_w)
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

    def build_network(self):
        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode

        bev_context_layer[0] = self.pModel.seq_num * rv_context_layer[0]    # 576 = 9 * 64

        # ----------------  network modify ----------------
        # self.point_pre1 = backbone.PointNetStacker(7+self.seq_num-1, rv_context_layer[0], pre_bn=True, stack_num=2)  # (15,64), 2*PointNet
        # self.bev_net = bird_view.BEVNet(bev_base_block, rv_context_layer, bev_layers, use_att=True)

        # network
        self.point_pre = backbone.PointNetStacker(7, rv_context_layer[0], pre_bn=True, stack_num=2)  # (7,64), 2*PointNet
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers, use_att=True)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, use_att=True)
        self.bev_grid2point = get_module(bev_grid2point, in_dim=self.bev_net.out_channels)
        self.rv_grid2point = get_module(rv_grid2point, in_dim=self.rv_net.out_channels)

        point_fusion_channels = (rv_context_layer[0], self.bev_net.out_channels, self.rv_net.out_channels)
        self.point_post = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels) # [64,64,64],64

        # MOS-head
        self.pred_layer = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)
        self.pred_bev_layer = nn.Sequential(
            nn.Conv2d(rv_context_layer[0] + self.bev_net.out_channels, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            backbone.PredBranch(64, self.pModel.class_num)
        )

        # Semantic-head
        self.pred_layer_single = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num_single)
        self.pred_bev_layer_single = nn.Sequential(
            nn.Conv2d(rv_context_layer[0] + self.bev_net.out_channels, 64, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            backbone.PredBranch(64, self.pModel.class_num_single)
        )


    # input:(1,9,7,130000,1),(1,9,130000,3,1),(1,9,130000,2,1)
    # infer:(4,9,7,160000,1),(4,9,160000,3,1),(4,9,160000,2,1)
    def stage_forward(self, point_feat, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            point_feat (BS, T, C, N, 1), C -> (x,y,z,???)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            point_feat_out (BS, C1, N, 1)
        '''
        BS, T, C, N, _ = point_feat.shape                               # (1,9,7,130000)

        pcds_cood_cur = pcds_coord[:, 0, :, :2].contiguous()            # (1,130000,2,1)
        pcds_sphere_coord_cur = pcds_sphere_coord[:, 0].contiguous()    # (1,130000,2,1)

        # BEV branch
        point_feat_tmp = self.point_pre(point_feat.view(BS*T, C, N, 1))                 # 特征的最大池化
        bev_input = VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0)) #(BS*T, C, H, W)
        bev_input = bev_input.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])      # [1, 9*64, 512, 512]
        # 初始化一个列表来存储特征残差
        residuals = []
        features_per_frame = 64
        # 遍历1-8帧，计算与第0帧的特征残差
        # 提取第0帧的特征
        base_features = bev_input[:, 0 * features_per_frame:1 * features_per_frame, :, :]
        for i in range(1, self.seq_num):  # 从第1帧到第8帧
            # 提取第i帧的特征
            frame_features = bev_input[:, i * features_per_frame:(i + 1) * features_per_frame, :, :]
            # 计算特征残差
            residual = frame_features - base_features
            # 将残差添加到列表中
            residuals.append(residual)

        # 将残差列表转换为张量
        residuals = torch.stack(residuals, dim=1)
        bev_concat_features = torch.cat((base_features.unsqueeze(1), residuals), dim=1)
        bev_concat_features = bev_concat_features.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])
        ic(base_features.unsqueeze(1).shape, residuals.shape, bev_concat_features.shape)
        bev_feat = self.bev_net(bev_concat_features)                                  # MAFL
        point_bev_feat = self.bev_grid2point(bev_feat, pcds_cood_cur)
        ic(point_bev_feat.shape)

        # ***********************************************************  BEV 分支  ********************************************************
        # # BEV [输入：增加x,y,z; ]
        # # 网络模型：增加 MGA+CAG+MCM+ASPP 模块
        # point_feat_tmp = self.point_pre(point_feat.view(BS*T, C, N, 1))        # (1,9,7,130000,1),(1*9,7,13000,1) ----> [BS*T,64,N,1]:[9,64,130000,1]
        # # Polarnet+Cylinder3d(极坐标+柱面分区)
        # # ----------------------------------------------- point-to-grid(BEV):  ------------------------------------------
        # # (BS*T, 7, N, 1),(BS*T, N, 2(x_quan,y_quan), 1) ---> (BS*T, C(7), 512, 512),   bev_input:(9,7,512,512)
        # bev_input = VoxelMaxPool(pcds_feat=point_feat.view(BS*T, C, N, 1), pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        # # (BS*T, 1(z), N, 1),(BS*T, N, 2(x_quan,y_quan), 1) ---> (BS*T, z(1), 512, 512), bev_input_res:(9,1,512,512)
        # bev_input_res = VoxelMaxPool(pcds_feat=point_feat.view(BS*T, C, N, 1)[:,2:3].contiguous(), pcds_ind=pcds_coord.view(BS*T, N, 3, 1)[:, :, :2].contiguous(), output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        #
        #
        # # (1*9(BS*T),7,512,512)--->(1(BS),63(T*C),512,512)
        # bev_input = bev_input.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])              # [BS*T, C(7), 512, 512] ---> [BS, T*C(7), 512, 512]
        # # (1*9(BS*T),1,512,512)--->(1(BS),9(T*C),512,512)
        # bev_input_res = bev_input_res.view(BS, -1, self.bev_wl_shape[0], self.bev_wl_shape[1])      # [BS*T, C(1), 512, 512] ---> [BS, T*C(1), 512, 512]
        #
        # # 如何计算BEV图像的残差???
        # C_bev = 7      # 特征残差=64/高度残差=1
        # bev_input_seven = bev_input[:,:C_bev]                   # 当前帧BEV图像:[1,7,512,512]
        #
        # # 如何计算BEV图像的残差???
        # C_bev_res = 1      # 特征残差=64/高度残差=1
        # bev_input_0 = bev_input_res[:,:C_bev_res]               # 当前帧BEV高度图像:[1,1,512,512]
        # bev_input_residual=[]
        # for i in range(1, self.seq_num):                                # [1,9):[1,2,...,8]
        #     input_res = bev_input_res[:,(i)*C_bev_res:(i+1)*C_bev_res]  # input_res:(1,1,512,512)
        #     residual = torch.abs(bev_input_0-input_res)                 # 之前帧的BEV残差图像 = 当前帧BEV高度图像 - 之前帧BEV的高度图像
        #     bev_input_residual.append(residual)
        #
        # bev_input_residual = torch.cat((bev_input_residual),dim=1)          # bev_input_residual:(1,8,512,512)
        # bev_input = torch.cat((bev_input_seven,bev_input_residual),dim=1)   # 当前帧的BEV图像 + 之前帧的BEV残差图像 = bev_input：(1, 7+8, 512, 512)
        #
        # bev_input = self.point_pre1(bev_input)              # (1, 15(7+8), 512, 512)--->(1,64,512,512)
        # # input:(1,64,512,512), output:(1,64,256,256)
        # bev_feat = self.bev_net(bev_input)                  # 3D稀疏卷积:(1,64,256,256)
        # point_bev_feat = self.bev_grid2point(bev_feat, pcds_cood_cur)   # BEV特征(1,64,256,256)+坐标(1,130000,2,1)--->(1,64,130000,1)
        #                                                                 # (BS, channel(64), 512, 512),(BS, N, 2(x,y), 1) ----> (BS,channel(64),N,1)


        # ---------------------------------------------------------- range-view 分支------------------------------------------------------------
        point_feat_tmp_cur = point_feat_tmp.view(BS, T, -1, N, 1)[:, 0].contiguous()        # (1,64,130000,1)
        # input:(1,64,130000,1),(1,130000,2(vertical_quan, horizon_quan),1)  output:(1,64,64,2048)
        rv_input = VoxelMaxPool(pcds_feat=point_feat_tmp_cur, pcds_ind=pcds_sphere_coord_cur, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        # input:(1,64,64,2048), output:(1,64,64,1024)
        rv_feat = self.rv_net(rv_input)                                     # rv_feat:(1,64,64,1024)
        point_rv_feat = self.rv_grid2point(rv_feat, pcds_sphere_coord_cur)  # RV特征(1,64,64,1024)+坐标(1,130000,2,1)--->point_rv_feat:(1,64,130000,1)

        # merge multi-view
        ic(point_feat_tmp_cur.shape, point_bev_feat.shape, point_rv_feat.shape)
        point_feat_out = self.point_post(point_feat_tmp_cur, point_bev_feat, point_rv_feat)  # 3*(1,64,130000,1) ---> (1,64,130000,1)
        point_feat_out_bev = torch.cat((point_feat_tmp_cur, point_bev_feat), dim=1)          # 2*(1,64,130000,1) ---> (1,128,130000,1)

        # pred
        pred_cls = self.pred_layer(point_feat_out).float()              # pred_cls:(1,3,130000,1)
        pred_bev_cls = self.pred_bev_layer(point_feat_out_bev).float()  # pred_bev_cls:(1,3,130000,1)

        pred_single = self.pred_layer_single(point_feat_out).float()  # pred_cls:(1,3,130000,1)
        pred_bev_single = self.pred_bev_layer_single(point_feat_out_bev).float()  # pred_bev_cls:(1,3,130000,1)

        ic('stage_forward_output:', pred_cls.shape, pred_single.shape)
        ic('stage_forward_output:', pred_cls, pred_single)

        return pred_cls, pred_bev_cls, pred_single, pred_bev_single                               # output:(1,3,130000,1)*2

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):              # pred_cls, pred_cls_raw:(1,3,130000,1)
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)           # (1,3,130000,1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord,
                pcds_target, pcds_target_single,
                pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord, pcds_coord_raw (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_target (BS, N, 1)
        Output:
            loss
        '''
        pred_cls, pred_bev_cls, pred_single, pred_bev_single = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)                       # pred_cls,pred_bev_cls:2*(1,3,130000,1)
        pred_cls_raw, pred_bev_cls_raw, pred_single_raw, pred_bev_single_raw = self.stage_forward(pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)   # pred_cls_raw,pred_bev_cls_raw: 2*(1,3,130000,1)


        # ------------------------------- MOS head ---------------------------------------
        loss1 = self.criterion_seg_cate(pred_cls, pcds_target) + 2 * lovasz_softmax(pred_cls, pcds_target, ignore=0)    # pcds_target:(1,130000,1)
        loss2 = self.criterion_seg_cate(pred_cls_raw, pcds_target) + 2 * lovasz_softmax(pred_cls_raw, pcds_target, ignore=0)
        loss3 = self.consistency_loss_l1(pred_cls, pred_cls_raw)
        loss_bev1 = self.criterion_seg_cate(pred_bev_cls, pcds_target) + 2 * lovasz_softmax(pred_bev_cls, pcds_target, ignore=0)
        loss_bev2 = self.criterion_seg_cate(pred_bev_cls_raw, pcds_target) + 2 * lovasz_softmax(pred_bev_cls_raw, pcds_target, ignore=0)
        loss_mos = 0.5 * (loss1 + loss2) + loss3 + 0.5 * (loss_bev1 + loss_bev2)

        # ------------------------------- Semantic head ---------------------------------------
        loss1_single = self.criterion_seg_cate(pred_single, pcds_target_single) + 2 * lovasz_softmax(pred_single, pcds_target_single, ignore=0)  # pcds_target:(1,130000,1)
        loss2_single = self.criterion_seg_cate(pred_single_raw, pcds_target_single) + 2 * lovasz_softmax(pred_single_raw, pcds_target_single, ignore=0)
        loss3_single = self.consistency_loss_l1(pred_single, pred_single_raw)
        loss_bev1_single = self.criterion_seg_cate(pred_bev_single, pcds_target_single) + 2 * lovasz_softmax(pred_bev_single, pcds_target_single, ignore=0)
        loss_bev2_single = self.criterion_seg_cate(pred_bev_single_raw, pcds_target_single) + 2 * lovasz_softmax(pred_bev_single_raw, pcds_target_single, ignore=0)
        loss_single = 0.5 * (loss1_single + loss2_single) + loss3_single + 0.5 * (loss_bev1_single + loss_bev2_single)

        # ------------------------------------ Total loss ------------------------------
        # loss = loss_mos + 0.5 * loss_single                       # checkpoints_semantickitti
        # ic('forward_loss:', 'ratio=1:0.5', loss_mos, loss_single, loss)

        loss = loss_mos + loss_single                               # checkpoints_kitti_road
        ic('forward_loss:', 'ratio=1:1',  loss_mos, loss_single, loss)
        

        return loss

    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            pcds_xyzi (BS, T, C, N, 1), C -> (x, y, z, intensity, dist, ...)
            pcds_coord (BS, T, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, T, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_cls, (BS, C, N, 1)
        '''
        # print('mvf_infer')
        # input:pcds_xyzi:(4,9,7,160000,1), pcds_coord:(4,9,160000,3,1), pcds_sphere_coord:(4,9,160000,2,1)
        # output: pred_cls:(4,3,160000,1), pred_bev_cls:(4,3,160000,1)
        pred_cls, pred_bev_cls, pred_single, pred_bev_single = self.stage_forward(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        return pred_cls, pred_single                     # pred_cls:(4,3,160000,1)



if __name__ == "__main__":
    # 假设有 pModel 对象
    # 请根据实际的 pModel 实例化代码来调整
    pModel = None  # pModel 应该是你具体定义的模型结构

    # 创建 AttNet 网络实例
    model = AttNet(pModel)

    # 设置设备为 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义测试输入，假设 (BS, T, C, N, 1) 形状为 (1, 9, 7, 130000, 1)
    BS = 1  # Batch size
    T = 9  # 序列长度
    C = 7  # 特征维度
    N = 130000  # 点云数量

    # 随机生成输入特征数据
    point_feat = torch.randn((BS, T, C, N, 1)).to(device)
    pcds_coord = torch.randint(0, 512, (BS, T, N, 3, 1)).to(device)                 # xyz
    pcds_sphere_coord = torch.randint(0, 2048, (BS, T, N, 2, 1)).to(device)         # uv

    # 前向传播
    pred_cls, pred_bev_cls = model.stage_forward(point_feat, pcds_coord, pcds_sphere_coord)

    # 输出预测结果
    print(f"Pred Class Shape: {pred_cls.shape}")
    print(f"Pred BEV Class Shape: {pred_bev_cls.shape}")

