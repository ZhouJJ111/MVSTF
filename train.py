import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4"
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
import logging
import importlib
from utils.logger import config_logger
from utils import builder
from icecream import ic

#import torch.backends.cudnn as cudnn
#cudnn.deterministic = True
#cudnn.benchmark = False


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def train_fp16(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency):
    scaler = torch.cuda.amp.GradScaler()                # fp16:主要区别
    rank = torch.distributed.get_rank()
    model.train()
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
            pcds_target, pcds_target_single,
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw,
            meta_list) in tqdm.tqdm(enumerate(train_loader)):

        # ic(pcds_xyzi.size(), pcds_coord.size(),pcds_sphere_coord.size())              # [1,9,7,130000,1],[1,9,130000,3,1],[1,9,130000,2,1]
        # ic( pcds_target.size())                                                       # [1,130000,1]
        # ic(pcds_xyzi_raw.size(), pcds_coord_raw.size(),pcds_sphere_coord_raw.size())  # [1,9,7,130000,1],[1,9,130000,3,1],[1,9,130000,2,1]
        # ic(meta_list)                                                                 # list:0-8, [-8,-7,...,-1,0]

        with torch.cuda.amp.autocast():                 # fp16:主要区别
            loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord,
                         pcds_target, pcds_target_single,
                         pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)       # 返回损失

        optimizer.zero_grad()
        # fp16:主要区别
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        reduced_loss = reduce_tensor(loss)
        if (i % log_frequency == 0) and rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch,\
                i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'])
            
            string = string + '; loss: {}'.format(reduced_loss.item() / torch.distributed.get_world_size())
            logger.info(string)


def train(epoch, end_epoch, args, model, train_loader, optimizer, scheduler, logger, log_frequency):
    rank = torch.distributed.get_rank()
    model.train()
    for i, (pcds_xyzi, pcds_coord, pcds_sphere_coord,
            pcds_target, pcds_target_single,
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw,
            meta_list) in tqdm.tqdm(enumerate(train_loader)):
        loss = model(pcds_xyzi, pcds_coord, pcds_sphere_coord,
                     pcds_target, pcds_target_single,
                     pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        reduced_loss = reduce_tensor(loss)
        if (i % log_frequency == 0) and rank == 0:
            string = 'Epoch: [{}]/[{}]; Iteration: [{}]/[{}]; lr: {}'.format(epoch, end_epoch,\
                i, len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'])
            
            string = string + '; loss: {}'.format(reduced_loss.item() / torch.distributed.get_world_size())
            logger.info(string)


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()

    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    os.system('mkdir -p {}'.format(model_prefix))

    # start logging
    config_logger(os.path.join(save_path, "log.txt"))
    logger = logging.getLogger()

    # reset dist
    device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')      # 注意：--nproc_per_node=1与GPU数量要一致
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # reset random seed
    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # define dataloader
    train_dataset = eval('datasets.{}.DataloadTrain'.format(pDataset.Train.data_src))(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=pGen.batch_size_per_gpu,
                            shuffle=(train_sampler is None),
                            num_workers=pDataset.Train.num_workers,
                            sampler=train_sampler,
                            pin_memory=True)

    print("rank: {}/{}; batch_size: {}".format(rank, world_size, pGen.batch_size_per_gpu))

    # define model
    base_net = eval(pModel.prefix)(pModel)  # mvf_vfe.AttNet
    # load pretrain model
    pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(pModel.pretrain.pretrain_epoch))
    if os.path.exists(pretrain_model):
        print('load_pretrain_model!')
        # base_net.load_state_dict(torch.load(pretrain_model, map_location='cpu'), strict=False)   # modify

        # modify
        # 加载检查点
        checkpoint = torch.load(pretrain_model, map_location='cpu')
        # 获取模型的状态字典
        state_dict = base_net.state_dict()
        # 过滤掉不匹配的参数
        new_checkpoint = {}
        for key in checkpoint:
            if key in state_dict and checkpoint[key].shape == state_dict[key].shape:
                new_checkpoint[key] = checkpoint[key]
        # 加载过滤后的状态字典
        base_net.load_state_dict(new_checkpoint, strict=False)
                        
        logger.info("Load model from {}".format(pretrain_model))

    base_net = nn.SyncBatchNorm.convert_sync_batchnorm(base_net)
    model = torch.nn.parallel.DistributedDataParallel(base_net.to(device),
                                                    device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)

    # define optimizer
    optimizer = builder.get_optimizer(pOpt, model)

    # define scheduler
    per_epoch_num_iters = len(train_loader)
    scheduler = builder.get_scheduler(optimizer, pOpt, per_epoch_num_iters)

    if rank == 0:
        logger.info(model)
        logger.info(optimizer)
        logger.info(scheduler)

    # start training
    for epoch in range(pOpt.schedule.begin_epoch, pOpt.schedule.end_epoch):  # (0,48)
        train_sampler.set_epoch(epoch)
        if pGen.fp16:
            train_fp16(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger, pGen.log_frequency)
        else:
            train(epoch, pOpt.schedule.end_epoch, args, model, train_loader, optimizer, scheduler, logger, pGen.log_frequency)

        # save model
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(model_prefix, '{}-model.pth'.format(epoch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
