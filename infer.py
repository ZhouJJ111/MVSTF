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
from datasets import utils

import tqdm
import importlib
import torch.backends.cudnn as cudnn
from icecream import ic
import yaml

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

    file = open(os.path.join(save_path, 'infer_fp16_{}.txt'.format(rank)), 'a')
    ic(file)

    with torch.no_grad():
        runtime = []
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
                pcds_target, pcds_target_single,
                valid_mask_list, pad_length_list, meta_list_raw,
                seq_id, file_id, point_num) in tqdm.tqdm(enumerate(val_loader)):

            with torch.cuda.amp.autocast():         # 主要区别: FP16
                start = time.time()
                pred_mos, pred_single = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(),\
                    pcds_sphere_coord.squeeze(0).cuda())

                end = time.time()
                res = end - start
                runtime.append(res)
                print(f'per frame {res}')


            valid_point_num = point_num
            # mos
            pred_mos = F.softmax(pred_mos, dim=1)
            pred_mos = pred_mos.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()                                             # pcds_target:(160000), (1,N,1)->(N)
            criterion_cate.addBatch(pcds_target[:valid_point_num], pred_mos[:valid_point_num])

            # single_semantic
            pred_single = F.softmax(pred_single, dim=1)
            pred_single = pred_single.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()  # pred_cls:(160000,3),  (4,3,N,1)->(3,N,1)->(1,N,3)->(N,3)
            pcds_target_single = pcds_target_single[0, :, 0].contiguous()  # pcds_target:(160000), (1,N,1)->(N)
            criterion_cate_single.addBatch(pcds_target_single[:valid_point_num], pred_single[:valid_point_num])

            # -----------------------------------------------------  MOS_infer --------------------------------------
            pred_mos = pred_mos[:valid_point_num].argmax(dim=1)
            pred_mos = pred_mos.cpu().numpy()
            pred_mos = pred_mos.reshape((-1)).astype(np.int32)
            pred_mos = pred_mos & 0xFFFF
            pred_mos = utils.relabel(pred_mos, task_cfg_mos['learning_map_inv'])        # [0,1,2]<->[0,9,251]
            ic(seq_id, file_id, point_num)
            ic(i, pred_mos.shape, type(pred_mos))
            # ic(pred_mos)

            seq_id_str = seq_id[0] if isinstance(seq_id, list) else seq_id
            file_id_str = file_id[0] if isinstance(file_id, list) else file_id
            file_id_str_with_label = f"{file_id_str}.label"  # 后面加上 .label
            save_mos_path = os.path.join(save_path, 'val08', 'MOS_sequences', seq_id_str, file_id_str_with_label)
            os.makedirs(os.path.dirname(save_mos_path), exist_ok=True)
            pred_mos.tofile(save_mos_path)

            # -----------------------------------------------------  single_infer --------------------------------------
            pred_single = pred_single[:valid_point_num].argmax(dim=1)
            pred_single = pred_single.cpu().numpy()
            pred_single = pred_single.reshape((-1)).astype(np.int32)
            pred_single = pred_single & 0xFFFF
            pred_single = utils.relabel(pred_single, task_cfg_single['learning_map_inv'])        # [0,1,2]<->[0,9,251]
            ic(seq_id, file_id, point_num)
            ic(i, pred_single.shape, type(pred_single), pred_single)

            seq_id_str = seq_id[0] if isinstance(seq_id, list) else seq_id
            file_id_str = file_id[0] if isinstance(file_id, list) else file_id
            file_id_str_with_label = f"{file_id_str}.label"  # 后面加上 .label
            save_single_path = os.path.join(save_path, 'val08' ,'Single_sequences', seq_id_str, file_id_str_with_label)
            os.makedirs(os.path.dirname(save_single_path), exist_ok=True)
            pred_single.tofile(save_single_path)


        mean_runtime = np.mean(runtime)
        print(f'seq 08 mean runtime {mean_runtime}')
        metric_cate = criterion_cate.get_metric()
        metric_cate_single = criterion_cate_single.get_metric()

        string = 'Epoch_mos {}'.format(epoch)
        string_single = 'Epoch_single {}'.format(epoch)
        for key in metric_cate:
            string = string + '; ' + key + ': ' + str(metric_cate[key])
        for key in metric_cate_single:
            string_single = string_single + '; ' + key + ': ' + str(metric_cate_single[key])

        file.write(string + '\n')
        file.write(string_single + '\n')
        file.close()



def val(epoch, model, val_loader, category_list, save_path, rank=0):
    criterion_cate = MultiClassMetric(category_list)
    
    model.eval()
    f = open(os.path.join(save_path, 'record_{}.txt'.format(rank)), 'a')
    with torch.no_grad():
        for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_target, valid_mask_list, pad_length_list, meta_list_raw) in tqdm.tqdm(enumerate(val_loader)):
            pred_cls = model.infer(pcds_xyzi.squeeze(0).cuda(), pcds_coord.squeeze(0).cuda(),\
                pcds_sphere_coord.squeeze(0).cuda())
            
            pred_cls = F.softmax(pred_cls, dim=1)
            pred_cls = pred_cls.mean(dim=0).permute(2, 1, 0).squeeze(0).contiguous()
            pcds_target = pcds_target[0, :, 0].contiguous()
            
            valid_point_num = pcds_target.shape[0]
            criterion_cate.addBatch(pcds_target, pred_cls[:valid_point_num])
        
        #record segmentation metric
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
    model_prefix = os.path.join(save_path, "checkpoint_best")

    # save_path_prediction = os.path.join("experiments/prediction", prefix)

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
            model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
            if pGen.fp16:
                val_fp16(epoch + rank, model, val_loader, pGen.category_list, pGen.category_list_single, model_prefix, rank)
            else:
                val(epoch + rank, model, val_loader, pGen.category_list, model_prefix, rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
