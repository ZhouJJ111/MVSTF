# -*-coding:utf-8-*-
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
from datasets import utils
from torch.utils.data.distributed import DistributedSampler

import datasets

from utils.metric import MultiClassMetric
from models import *

import tqdm
import importlib
import torch.backends.cudnn as cudnn
from icecream import ic
import yaml

cudnn.benchmark = True
cudnn.enabled = True


def test_fp16(epoch, model, test_loader, category_list, save_path_prediction, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    print('FP16 inference mode!')
    model.eval()

    with open('datasets/semantic-kitti.yaml', 'r') as f:
        task_cfg_mos = yaml.load(f)

    with torch.no_grad():
        runtime = []
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
                valid_mask_list, pad_length_list, meta_list_raw,
                point_num, seq_id, idx) in tqdm.tqdm(enumerate(test_loader)):

            with torch.cuda.amp.autocast():
                start = time.time()
                pred_cls_mos, pred_cls_semantic, pred_cls_multi = model.infer(pcds_xyzi.squeeze(0).cuda(),
                                                                              pcds_coord.squeeze(0).cuda(),
                                                                              pcds_sphere_coord.squeeze(0).cuda()
                                                                              )
                res = time.time() - start
                runtime.append(res)
                print(f'per frame {res}')

            #  MOS
            pred_cls_mos = F.softmax(pred_cls_mos, dim=1)
            pred_cls_mos = pred_cls_mos.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()


            pred_cls_mos = pred_cls_mos.argmax(dim=1)
            pred_cls_mos = pred_cls_mos.cpu().numpy()
            pred_cls_mos = pred_cls_mos.reshape((-1)).astype(np.int32)
            pred_cls_mos = pred_cls_mos & 0xFFFF
            learning_map_inv_mos = task_cfg_mos['learning_map_inv']  # 返回真实的标签
            pred_cls_mos = utils.relabel(pred_cls_mos, learning_map_inv_mos)

            # Ensure save_path_prediction exists and create necessary subdirectories
            save_dir = os.path.join(save_path_prediction, 'sequences', str(seq_id[0]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Construct the full file path
            file_name = f'{int(idx):06d}.label'
            mos_path = os.path.join(save_dir, file_name)

            pred_cls_raw_mos = pred_cls_mos[:point_num]  # 原始点云数
            ic('zjj:', point_num, pred_cls_raw_mos.shape)
            pred_cls_raw_mos.tofile(mos_path)


        mean_runtime = np.mean(runtime)
        mean_runtime2 = sum(runtime) / len(runtime)
        print(f'seq 08 mean runtime {mean_runtime} OR {mean_runtime2}')
        # record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)

        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])

        file = open(os.path.join(save_path_prediction, 'record_fp16_infer_{}.txt'.format(rank)), 'a')
        file.write(string + '\n')
        file.close()


def test(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)

    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, valid_mask_list, pad_length_list,
                meta_list_raw) in tqdm.tqdm(enumerate(val_loader)):
            pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(), \
                                   pcds_sphere_coord.squeeze(0).cuda())

            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()

            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])

        # record segmentation metric
        metric_cate = criterion_cate.get_metric()
        string = 'Epoch {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])

        f.write(string + '\n')
        f.close()


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    save_path_prediction = os.path.join("experiments/prediction_test", prefix)

    # reset dist
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # define dataloader
    test_dataset = eval('datasets.{}.DataloadTest'.format(pDataset.Test.data_src))(pDataset.Test)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=pDataset.Test.num_workers,
                             pin_memory=True)

    # define model
    model = eval(pModel.prefix)(pModel)
    model.cuda()
    model.eval()

    for epoch in range(args.start_epoch, args.end_epoch + 1, world_size):
        if (epoch + rank) < (args.end_epoch + 1):
            pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(epoch + rank))
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            if pGen.fp16:
                test_fp16(epoch + rank, model, test_loader, pGen.category_list, save_path_prediction, rank)
            else:
                test_fp16(epoch + rank, model, test_loader, pGen.category_list, save_path_prediction, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
