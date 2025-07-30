import _init_paths

import os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict

def save_images(sal, img_path, save_path, opt, stack=None):
    # print(img_path,save_path)
    tmp = img_path.split('/') 
    video_name='_'.join(tmp[-1].split('_')[:-1])
    os.makedirs(os.path.join(save_path, video_name), exist_ok=True)
    sal_name =  tmp[-1].replace('.jpg', '.png')
    # sal_name = tmp[-3] + '/' + tmp[-1][-9:].replace('.jpg', '.png')     # python_val index.png

    gt = Image.open(img_path)
    gt = np.asarray(gt, np.float32)
    # gt /= (gt.max() + 1e-8)
    
    sal = F.interpolate(sal, size=(gt.shape[0], gt.shape[1]), mode='bilinear', align_corners=False)
    sal = sal.sigmoid().data.cpu().numpy().squeeze()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    
    # For ZVOS
    # sal[np.where(sal>=0.5)] = 1
    # sal[np.where(sal<0.5)] = 0

    sal = (sal*255).astype(np.uint8)

    sal = Image.fromarray(sal)
    if stack is not None:
        sal=cv2.addWeighted(sal,0.5,cv2.resize(stack,sal.shape),0.5,0)
    sal.save(os.path.join(save_path , video_name, sal_name))

def rm_module(saved_state_dict):
    new_state_dict = OrderedDict()
    for k, v in saved_state_dict['network'].items():
        new_state_dict[k.replace('module.', '')] = v
    return new_state_dict



