import _init_paths

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import os.path as osp
from evaluator import Eval_thread
from dataloader import EvalDataset

def main(cfg):
    if cfg.methods is None:
        method_names = os.listdir(cfg.pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(cfg.gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(img_root=osp.join(cfg.pred_dir, dataset),
                                 label_root=osp.join(cfg.gt_dir, dataset, 'mask/val'),
                                 use_flow=config.use_flow)
            thread = Eval_thread(loader, method, dataset)
            threads.append(thread)
    for thread in threads:
        print(thread.run()[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='val_vsod')
    parser.add_argument('--methods', type=str, default='HMHI-Net',help='Evaluated File Name, eg:HMHI-Net. Can be a name list separated by a single space')  
    parser.add_argument('--datasets', type=str, default='DAVIS_byvideo',help='Evaluated Dataset Name. Can be a name list separated by a single space') 
    parser.add_argument('--gt_dir', type=str, default='/data/UVOS/',help='Evaluated Dataset Root') 
    parser.add_argument('--pred_dir', type=str, default='/output/',help='Evaluated File Root') 
    parser.add_argument('--use_flow', type=bool, default=True)

    config = parser.parse_args()
    
    config.methods='HMHI-Net'  
    config.datasets='FBMS DAVIS-16 YTBOBJ' 
    config.pred_dir='/output/'
    main(config)
