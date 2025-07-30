import os
import _init_paths
from config import Parameters
opt = Parameters().parse()

import time
import torch
from torch.utils import data
from lib.HMHI_Net import HMHI_Net
from utils.evaluator import Eval_thread
from utils.dataloader import EvalDataset
from utils.val_zvos import compute_J_F,eval_JF_from_data

from utils.dataloader import val_video_data_loader
from utils.init import load_model
from utils.func import save_images
from tqdm import tqdm


def demo(opt):

    model = HMHI_Net(opt)
    model = load_model(model, opt)

    model.cuda()
    model.eval()
    test_dataset_list = opt.infer_dataset
    if not isinstance(test_dataset_list,list):
        test_dataset_list = [test_dataset_list]
    for dataset in test_dataset_list:
        save_path = opt.infer_save + dataset + '/'

        test_dataset = val_video_data_loader(os.path.join(opt.infer_dataset_path, dataset), opt.img_size, data_augmentation=False, mode='val')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.val_batchsize, shuffle=False, num_workers=32, pin_memory=True)

        dataset_size = len(test_dataset)

        with torch.no_grad():
            img_count = 1
            time_total = 0
            for step, data_pack in tqdm(enumerate(test_loader)):

                images, flows, img_paths = data_pack
                images = images.cuda()
                flows = flows.cuda()

                time_start = time.perf_counter()
                preds  = model(images, flows, mode='val')
                cur_time = (time.perf_counter() - time_start)

                time_total += cur_time
                
                for index in range(len(preds)):
                    sal = preds[index]
                    save_images(sal, img_paths[index][0], save_path, opt)

                    img_count += 1
            
            print("\n[INFO-Test-Done] FPS: {}".format(dataset_size / time_total))
        if  opt.eval_vsod and dataset in ['DAVSOD2SEG_byvideo','ViSal2SEG_byvideo','DAVIS2SEG_byvideo','FBMS2SEG_byvideo']:
            gt_path = os.path.join(opt.infer_dataset_path, dataset, 'mask/val')
            loader = EvalDataset(img_root=save_path, label_root=gt_path, use_flow=True)
            thread = Eval_thread(loader, dataset, dataset)
            measure_results = thread.run(0, 0)
            
            print(dataset + '_MAE', measure_results[0][0],)
            print(dataset + '_Max_Fmeasure', measure_results[0][1])
            print(dataset + '_Max_Emeasure', measure_results[0][2])
            print(dataset + '_Max_Smeasure', measure_results[0][3])

        if not opt.eval_vsod and dataset in ['DAVIS2SEG_byvideo', 'FBMS2SEG_byvideo', 'YTBOBJ2SEG_byvideo']:
            gt_path = os.path.join(opt.infer_dataset_path, dataset, 'mask/val')
            J_mean, F_mean = eval_JF_from_data(epoch=0, total_epoch=0, test_dataset=dataset, gt_path=gt_path, pred_path=save_path)
            JF_mean = (J_mean + F_mean) / 2
            print(f'Inference at {dataset}, J: {J_mean}, F:{F_mean}, J&F:{JF_mean}')

if __name__ == "__main__":
    demo(opt=opt)
