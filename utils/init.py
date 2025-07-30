import _init_paths

import os
import torch
import random
import numpy as np
from torch.utils import data
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from datetime import timedelta

from utils.dataloader import train_video_data_loader, val_video_data_loader
from utils.func import rm_module
import datetime

def init_seed(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    cudnn.benchmark = False

def init_tensorboard(opt):
    os.makedirs(opt.save_path, exist_ok=True)
    writer = SummaryWriter(opt.save_path)
    return writer

def init_device(opt):
    # device setting
    device = torch.device('cuda:{}'.format(opt.local_rank))
    torch.cuda.set_device(device)
    
    timeout_long_ncll = timedelta(seconds=6000)
    dist.init_process_group(backend='nccl', timeout=timeout_long_ncll)

    num_gpus = torch.cuda.device_count() 
    if opt.local_rank == 0: print('Let us use', num_gpus, 'GPUs!')

    torch.cuda.empty_cache()

def init_ddp_model(model, opt):
    model = torch.nn.parallel.DistributedDataParallel(model.to(opt.local_rank),  device_ids=[opt.local_rank], find_unused_parameters=True)
    return model

def init_loss_func(opt):
    BCE = torch.nn.BCEWithLogitsLoss().to(opt.local_rank)
    return BCE

def init_scheduler(opt, optimizer):
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma = 0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=70,eta_min=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9,last_epoch=-1)
    return scheduler


def init_optimizer(model, opt, data_loader):
    # model.cuda() must be placed before optimizer init.
    if 'swin' in opt.encoder:
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if 'absolute_pos_embed' in name:
                no_decay.append(param)
            elif 'relative_position_bias_table' in name:
                no_decay.append(param)
            elif 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        parameters = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.}]
        optimizer = torch.optim.AdamW(parameters, opt.lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        if opt.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
        elif opt.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), opt.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    
    return optimizer

def init_train_dataloader(opt):
    if isinstance(opt.train_dataset,list):
        
        assert len(opt.train_dataset_repeat_list) <= len(opt.train_dataset),\
            f"train_dataset_repeat_list length { len(opt.train_dataset_repeat_list)} should be smaller than train_dataset length {len(opt.train_dataset)}"
        repeat_list = opt.train_dataset_repeat_list
        if len(opt.train_dataset_repeat_list) != len(opt.train_dataset):
            for i in range(len(opt.train_dataset)-len(opt.train_dataset_repeat_list)):
                repeat_list.append(1)
                
        all_dataset=[]
        for train_dataset, repeat_times in zip(opt.train_dataset, repeat_list):
            print(f"Constructing Training Dataset {train_dataset}, repeat {repeat_times} times")
            for i in range(repeat_times):
                temp_dataset=train_video_data_loader(os.path.join(opt.train_root,train_dataset), opt.img_size, opt.data_augmentation, mode='train',select_num = opt.train_num_frame)
                all_dataset.append(temp_dataset)
        dataset=data.ConcatDataset(all_dataset)
    else:
        raise Exception(f"Wrong type for opt.train_dataset")
    
    sampler = data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset=dataset,
                                  sampler=sampler,
                                  batch_size=opt.train_batchsize,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    iters_pre_epoch = len(data_loader)
    total_iters = opt.epoch*iters_pre_epoch
    return data_loader, sampler, iters_pre_epoch, total_iters

def init_val_dataloader(opt):
    if isinstance(opt.val_dataset,list):
        val_data_loader=[]
        for val_dataset in opt.val_dataset:
            temp_dataset=val_video_data_loader(os.path.join(opt.val_root, val_dataset), opt.img_size, data_augmentation=False, mode='val')
            temp_val_dataloader=data.DataLoader(dataset=temp_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
            val_data_loader.append(temp_val_dataloader)
    else:
        raise Exception(f"Wrong type for opt.train_dataset")
    return val_data_loader

def load_model(model, opt):
    print(opt.infer_model_path)
    pretrain = torch.load(opt.infer_model_path, map_location='cpu')
    if len(opt.gpu) > 1:
        # for the multiple gpus
        gpu_list = []
        for i in opt.gpu:
            if i != ',': gpu_list.append(int(i)-int(opt.gpu[0]))
            pass
        model = torch.nn.DataParallel(model, device_ids=gpu_list)
        model.load_state_dict(pretrain['network'], strict=True)
    else:
        # for a single gpu
        new_state_dict = rm_module(pretrain)
        model.load_state_dict(new_state_dict, strict=True)
    return model

def resume_model_optimizer(model, optimizer, opt, data_loader):
    saved_state_dict = torch.load(opt.restore_from, map_location='cpu') 
    
    start_epoch = saved_state_dict['epoch']
    model.load_state_dict(saved_state_dict['network'])
    print(optimizer)
    optimizer.load_state_dict(saved_state_dict['optimizer'])
    if opt.local_rank==0: 
        print('Successfully resume model from', opt.restore_from)
    if opt.lr != optimizer.param_groups[0]['lr']:
        if opt.local_rank==0: 
            print('Successfully change learning rate from ', optimizer.param_groups[0]['lr'], ' to ', opt.lr)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, betas=(0.9, 0.999), weight_decay=5e-4)

    return start_epoch, model, optimizer
