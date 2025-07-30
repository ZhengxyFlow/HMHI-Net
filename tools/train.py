# This file is modified based on https://github.com/DLUT-yyc/Isomer (OriginalProject)
# Original license: MIT
# Modified 2025
import _init_paths

import torch
from config import Parameters
opt = Parameters().parse()

from lib.HMHI_Net import HMHI_Net

from utils.init import *
from utils.apis_multi import *
import json
import time


def main(opt):

    writer = init_tensorboard(opt)
    
    with open(os.path.join(opt.save_path,'all_args.json'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    init_seed(opt)
    init_device(opt)

    data_loader, sampler, iters_pre_epoch, total_iters = init_train_dataloader(opt)

    if opt.val_dataset != 'None':
        all_val_data_loader = init_val_dataloader(opt)

    model = HMHI_Net(opt)
    
    with open(os.path.join(opt.save_path,'model.txt'), 'w') as f:  
        print(model, file=f)

    model = init_ddp_model(model, opt)
    
    optimizer = init_optimizer(model, opt, data_loader) 

    loss_func = init_loss_func(opt)
    scheduler = init_scheduler(opt, optimizer)

    start_epoch = 0
    if opt.restore_from != 'None':
        start_epoch, model, optimizer = resume_model_optimizer(model, optimizer, opt, data_loader)
    elif opt.load_from != 'None':
        saved_state_dict = torch.load(opt.load_from, map_location='cpu') 
        z=model.load_state_dict(saved_state_dict['network'],strict=False)
        print(f"Successfully Load From {opt.load_from}, with keys loaded: {z}")

    iters = start_epoch*iters_pre_epoch
    
    measure_socre={}
    for val_dataset in opt.val_dataset:
        measure_socre[val_dataset] = 0
        
    # training
    for epoch in range(start_epoch, opt.epoch):
        epoch += 1
        start =time.time()
        iters = train_one_epoch(model, loss_func, optimizer, data_loader, sampler, writer, epoch, iters, total_iters, iters_pre_epoch, opt)
        # scheduler.step()
        end =time.time()
        if opt.local_rank == 0:
            writer.add_scalar('train_each_epoch_time', end-start, epoch)
        
        UVOS_test = True
        SOD_test  = False
        ###### Save And Val ##########
        with torch.no_grad():
            if epoch % opt.val_every_epoch == 0:
                if (opt.val_dataset != 'None') & (opt.local_rank==0):
                    if isinstance(opt.val_dataset,list):
                        for idx in range(len(opt.val_dataset)):
                            val_dataset=opt.val_dataset[idx]
                            val_data_loader=all_val_data_loader[idx]
                            measure_score = val_one_epoch(model, val_data_loader, writer, epoch, opt, val_dataset)
                            if UVOS_test & (opt.save_model == True) & (measure_score >= measure_socre[val_dataset]):
                                save_model_optimizer(model, optimizer, epoch, opt, save_best=True, 
                                                    score=str(measure_score), 
                                                    last_score=str(measure_socre[val_dataset]),
                                                    val_dataset=val_dataset)
                                measure_socre[val_dataset] = measure_score
                            if SOD_test & (opt.save_model == True) & (measure_score >= measure_socre[val_dataset]):
                                save_model_optimizer(model, optimizer, epoch, opt, save_best=True, 
                                                    score=str(measure_score), 
                                                    last_score=str(measure_socre[val_dataset]),
                                                    val_dataset=val_dataset)
                                measure_socre[val_dataset] = measure_score
                            
        if (epoch % opt.save_every_epoch == 0) & (opt.save_model == True) & (opt.local_rank==0):
                save_model_optimizer(model, optimizer, epoch, opt, save_best=False)
        
        #save the latest epoch
        save_model_optimizer(model, optimizer, epoch, opt, save_best=True, val_dataset=None)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(opt)


