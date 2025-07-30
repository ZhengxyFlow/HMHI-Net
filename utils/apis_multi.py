import _init_paths

import os
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.autograd import Variable

from utils.func import *
import torch.nn.functional as F
from utils.evaluator import Eval_thread
from utils.dataloader import EvalDataset
from utils.val_zvos import compute_J_F,eval_JF_from_data
import math
from lib.utils.loss_func import *

def train_one_epoch(model, loss_func, optimizer, data_loader, sampler, writer, epoch, iters, total_iters, iters_pre_epoch, opt):
    model.train()
    optimizer.zero_grad()
    sampler.set_epoch(epoch)
    # ---- multi-scale training ----
    size_rates = [0.75,1,1.25] if opt.ms_train == True else [1]

    loop = tqdm(enumerate(data_loader, start=1), total =len(data_loader))
    for i, pack in loop:
        iters += 1
        for rate in size_rates:
            # ---- get data ----
            images, flows, gts, _img_path = pack
            images = Variable(images.cuda()) # BxTxCxHxW
            flows = Variable(flows.cuda())
            gts = Variable(gts.cuda())
            # ---- multi-scale training ----
            img_size = int(round(opt.img_size*rate/32)*32)

            if rate != 1:
                images = F.interpolate(images, size=(img_size, img_size), mode='bilinear', align_corners=False)
                gts = F.interpolate(gts, size=(img_size, img_size), mode='nearest')

            # ---- forward ----
            preds = model(images, flows, mode='train')
            # ---- cal loss ----
            
            dice_loss=Diceloss()
            focal_loss=FocalLoss()
            loss = 0.0
            loss_weight = 1
            
            assert len(preds) == gts.shape[1],\
                f"Input frame num {gts.shape[1]} ! output mask num {len(preds)}"
            
            N = len(preds)-1
            for i in range(len(preds)):
                # loss_ = loss_func(preds[i], gts)
                loss_ = 20*focal_loss(preds[i], gts[:,i])+dice_loss(preds[i], gts[:,i])
                loss +=  math.pow(loss_weight, N-i) * loss_
                
            # ---- backward ----
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            total_loss = loss / dist.get_world_size()

        if opt.local_rank == 0:
            writer.add_scalar('train_loss', total_loss, iters)
        loop.set_description(f'Epoch [{epoch}/{opt.epoch}] Training   [Video]')
        loop.set_postfix(loss = total_loss.item(), lr = optimizer.param_groups[0]['lr'])

    return iters

def val_one_epoch(model, val_data_loader, writer, epoch, opt, val_dataset):
    model.eval()
    with torch.no_grad():
        img_save_path = os.path.join(opt.infer_save, 'epoch_'+str(epoch), val_dataset)
        loop = tqdm(enumerate(val_data_loader), total =len(val_data_loader))
        for index, batch in loop:
            img, flow, img_paths = batch
            preds = model.module(Variable(img.cuda()), Variable(flow.cuda()), mode='val')
            assert img.size()[1] == len(preds),\
                f"output masks num len(preds) shoud equal input image num {img.size()[1]}"
            for bs_index in range(len(preds)):
                sal = preds[bs_index]
                save_images(sal, img_paths[bs_index][0], img_save_path, opt)
            loop.set_description(f'Epoch [{epoch}/{opt.epoch}] Inference  [{val_dataset}]')
            
    val_VSOD = opt.eval_vsod
    if val_VSOD and val_dataset in ['DAVSOD2SEG_byvideo','ViSal2SEG_byvideo','DAVIS2SEG_byvideo','ViSal','FBMS2SEG_byvideo']:
        gt_path = os.path.join(opt.val_root, val_dataset, 'mask/val')
        loader = EvalDataset(img_root=img_save_path, label_root=gt_path, use_flow=True)
        thread = Eval_thread(loader, opt.train_dataset, val_dataset)
        measure_results = thread.run(epoch, opt.epoch)
        writer.add_scalar(val_dataset + '_MAE', measure_results[0][0], epoch)
        writer.add_scalar(val_dataset + '_Max_Fmeasure', measure_results[0][1], epoch)
        writer.add_scalar(val_dataset + '_Max_Emeasure', measure_results[0][2], epoch)
        writer.add_scalar(val_dataset + '_Max_Smeasure', measure_results[0][3], epoch)
        
        print(val_dataset + '_MAE', measure_results[0][0], epoch)
        print(val_dataset + '_Max_Fmeasure', measure_results[0][1], epoch)
        print(val_dataset + '_Max_Emeasure', measure_results[0][2], epoch)
        print(val_dataset + '_Max_Smeasure', measure_results[0][3], epoch)
        return measure_results[0][3]
    if val_dataset in ['DAVIS2SEG_byvideo', 'FBMS2SEG_byvideo', 'YTBOBJ2SEG_byvideo']:
        gt_path = os.path.join(opt.val_root, val_dataset, 'mask/val')
        J_mean, F_mean = eval_JF_from_data(epoch=epoch, total_epoch=opt.epoch, test_dataset=val_dataset, gt_path=gt_path, pred_path=img_save_path)
        writer.add_scalar(val_dataset + '_J', J_mean, epoch)
        writer.add_scalar(val_dataset + '_F', F_mean, epoch)
        JF_mean = (J_mean + F_mean) / 2
        writer.add_scalar(val_dataset + '_J&F', JF_mean, epoch)
        print(f'Training at epoch_{epoch}/{opt.epoch}, eval at {val_dataset}, J: {J_mean}, F:{F_mean}, J&F:{JF_mean}')
        if val_dataset in ['DAVIS_byvideo']:
            return JF_mean
        else:
            return J_mean
    return 0
    
def save_model_optimizer(model, optimizer, epoch, opt, save_best, score='', last_score='', val_dataset=None):
    os.makedirs(opt.save_path, exist_ok=True)
    state = {'epoch':epoch, 'network':model.state_dict(), 'optimizer':optimizer.state_dict()}
    if save_best:
        if val_dataset is None:
            torch.save(state, opt.save_path + 'latest.pth')
        else:
            last_path = opt.save_path + f'{last_score}{val_dataset}_best.pth'
            if os.path.exists(last_path):
                os.remove(last_path)
            torch.save(state, opt.save_path + f'{score}{val_dataset}_best.pth')
    else:
        torch.save(state, opt.save_path + '{}epoch.pth'.format(epoch))
