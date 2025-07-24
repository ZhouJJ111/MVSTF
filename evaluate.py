import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn

import yaml
from icecream import ic

cudnn.benchmark = True
cudnn.enabled = True


def val_fp16(epoch, model, val_loader, category_list, category_list_single, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    criterion_cate_single = MultiClassMetric(category_list_single)
    print('FP16 inference mode!')
    model.eval()
    with open('datasets/semantic-kitti-mos.yaml', 'r') as f:
        task_cfg_mos = yaml.load(f)
    with open('datasets/semantic-kitti-single.yaml', 'r') as f:
        task_cfg_single = yaml.load(f)

    file = open(os.path.join(save_path, 'record_fp16_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        runtime = []
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
                pcds_target, pcds_target_single,
                valid_mask_list, pad_length_list, meta_list_raw,
                seq_id, idx, point_num) in tqdm.tqdm(enumerate(val_loader)):
            # pcds_xyzi:(1,4,9,7,160000,1), pcds_coord:(1,4,9,160000,3,1), pcds_sphere_coord:(1,4,9,160000,2,1),                 # 实现?具体各个参数的含义?
            # pcds_target:(1,160000,1)
            # valid_mask_list:(1,N)(list=9,False/True), pad_length_list:(1,), meta_list_raw:路径000000.label-000008.label

            with torch.cuda.amp.autocast():         # 主要区别: FP16
                # input:(4,9,7,160000,1),(4,9,160000,3,1),(4,9,160000,2,1)  output:(4,3,160000,1)
                start = time.time()
                pred_mos, pred_single = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(),\
                    pcds_sphere_coord.squeeze(0).cuda())

                end = time.time()
                res = end - start
                runtime.append(res)
                print(f'per frame {res}')

            # ic(pred_mos, pred_single)
            valid_point_num = point_num
            # mos
            pred_mos = F.softmax(pred_mos, dim=1)                                       # [4, 3, 160000, 1]
            pred_mos = pred_mos.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()    # pred_cls:(160000,3),  (4,3,N,1)->(3,N,1)->(1,N,3)->(N,3)
            pcds_target = pcds_target[0, :, 0].contiguous()                             # pcds_target:(160000), (1,N,1)->(N)
            ic(i, point_num)
            # ic(pcds_target[:valid_point_num].shape,  pred_mos[:valid_point_num].shape)
            # ic(pcds_target[:valid_point_num], pred_mos[:valid_point_num])
            criterion_cate.addBatch(pcds_target[:valid_point_num], pred_mos[:valid_point_num])

            # single_semantic
            pred_single = F.softmax(pred_single, dim=1)
            pred_single = pred_single.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()  # pred_cls:(160000,3),  (4,3,N,1)->(3,N,1)->(1,N,3)->(N,3)
            pcds_target_single = pcds_target_single[0, :, 0].contiguous()  # pcds_target:(160000), (1,N,1)->(N)
            # ic(pcds_target_single[:valid_point_num].shape, pred_single[:valid_point_num].shape)
            criterion_cate_single.addBatch(pcds_target_single[:valid_point_num], pred_single[:valid_point_num])


        mean_runtime = np.mean(runtime)
        print(f'seq 08 mean runtime {mean_runtime} ')
        #record segmentation metric
        metric_cate = criterion_cate.get_metric()
        metric_cate_single = criterion_cate_single.get_metric()

        string = 'Epoch_mos {}'.format(epoch)
        string_single = 'Epoch_single {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        for key in metric_cate_single:
            string_single = string_single + '; ' + key + ': ' + str(metric_cate_single[key])

        file.write(string + '\n')
        # file.write(string_single + '\n')
        file.close()


def val(epoch, model, val_loader, category_list, category_list_single, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    criterion_cate_single = MultiClassMetric(category_list_single)

    with open('datasets/semantic-kitti-mos.yaml', 'r') as f:
        task_cfg_mos = yaml.safe_load(f)
    with open('datasets/semantic-kitti-single.yaml', 'r') as f:
        task_cfg_single = yaml.safe_load(f)

    file = open(os.path.join(save_path, 'record_fp32_{}.txt'.format(rank)), 'a')
    model.eval()
    with torch.no_grad():
        runtime = []
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
                pcds_target, pcds_target_single,
                valid_mask_list, pad_length_list, meta_list_raw,
                seq_id, idx, point_num) in tqdm.tqdm(enumerate(val_loader)):
            # 确保所有张量都在正确的设备上，并且使用 FP32
            pcds_xyzi = pcds_xyzi.squeeze(0).float().cuda()
            pcds_coord = pcds_coord.squeeze(0).float().cuda()
            pcds_sphere_coord = pcds_sphere_coord.squeeze(0).float().cuda()

            start = time.time()
            pred_mos, pred_single = model.infer(pcds_xyzi, pcds_coord, pcds_sphere_coord)
            end = time.time()
            res = end - start
            runtime.append(res)
            print(f'per frame {res}')

            valid_point_num = point_num

            # mos
            pred_mos = F.softmax(pred_mos, dim=1)  # [4, 3, 160000, 1]
            pred_mos = pred_mos.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()  # pred_cls:(160000,3)
            pcds_target = pcds_target[0, :, 0].contiguous()  # pcds_target:(160000)
            criterion_cate.addBatch(pcds_target[:valid_point_num], pred_mos[:valid_point_num])

            # single_semantic
            pred_single = F.softmax(pred_single, dim=1)
            pred_single = pred_single.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()  # pred_cls:(160000,3)
            pcds_target_single = pcds_target_single[0, :, 0].contiguous()  # pcds_target:(160000)
            criterion_cate_single.addBatch(pcds_target_single[:valid_point_num], pred_single[:valid_point_num])

        mean_runtime = np.mean(runtime)
        print(f'seq 08 mean runtime {mean_runtime}')

        # 记录分割指标
        metric_cate = criterion_cate.get_metric()
        metric_cate_single = criterion_cate_single.get_metric()

        string = 'Epoch_mos '
        string_single = 'Epoch_single '
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        for key in metric_cate_single:
            string_single = string_single + '; ' + key + ': ' + str(metric_cate_single[key])

        file.write(string + '\n')
        file.write(string_single + '\n')
        file.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")                # modify

    # reset dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')      # 注意：--nproc_per_node=1与GPU数量要一致
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # define dataloader
    val_dataset = eval('datasets.{}.DataloadVal'.format(pDataset.Val.data_src))(pDataset.Val)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=pDataset.Val.num_workers,
                            pin_memory=True)
    
    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()
    
    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            ic('zjj:',  pretrain_model)
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            if pGen.fp16:
                val_fp16(epoch + rank, model, val_loader, pGen.category_list, pGen.category_list_single, model_prefix, rank)
                # val(epoch + rank, model, val_loader, pGen.category_list, pGen.category_list_single, model_prefix, rank)
            else:
                val(epoch + rank, model, val_loader, pGen.category_list, pGen.category_list_single, model_prefix, rank)
                # val_fp16(epoch + rank, model, val_loader, pGen.category_list, pGen.category_list_single, model_prefix, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
