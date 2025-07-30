import _init_paths

import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
from utils.data_augmentation import *
from glob import glob
import random

class train_video_data_loader(data.Dataset):
    def __init__(self, root, img_size, data_augmentation=False, mode='train', select_num=3):
        self.img_size = img_size
        self.data_augmentation = data_augmentation
        self.images, self.flows, self.gts = [], [], []
        self.num_frames= select_num
        
        img_path=os.path.join(root, 'frame', mode)
        gt_path=os.path.join(root, 'mask', mode)
        images_seq =[]
        gts_seq =[]
        for seq_name in os.listdir(img_path):
            seq_img = os.path.join(img_path, seq_name)
            # seq_flow = os.path.join(img_path, seq_name)
            seq_gt = os.path.join(gt_path, seq_name)
            if os.path.isdir:
                images_seq += [seq_img]
            if os.path.isdir:
                gts_seq += [seq_gt]
        
        self.images = sorted(images_seq)
        self.gts = sorted(gts_seq)
        
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),  # bilinear
            transforms.ToTensor()])

        if self.data_augmentation:
            # self.resize = Resize(img_scale=(int(self.img_size*4), self.img_size), ratio_range=(0.5,2.0))
            # self.crop = RandomCrop(crop_size=(self.img_size, self.img_size), cat_max_ratio=0.8)
            self.flip = RandomFlip(prob=0.5)
            self.pmd = PhotoMetricDistortion()
            # self.pad = Pad(size=(self.img_size, self.img_size), pad_val=0, seg_pad_val=0)

        self.size = len(self.gts)

    def augmentation(self, result):
        # result = self.resize(result)
        # result = self.crop(result)
        result = self.flip(result)
        result = self.pmd(result)
        # result = self.pad(result)

        return result

    def __getitem__(self, index):
        img_path = self.images[index]
        # flow_path = self.flows[index]
        gt_path = self.gts[index]
        flow_path = img_path.replace('frame', 'flow')
        
        assert os.path.isdir(img_path) and os.path.isdir(gt_path) and os.path.isdir(flow_path),\
            f'all paths {img_path} should be a path'
            
        assert os.path.exists(img_path) and os.path.exists(gt_path) and os.path.exists(flow_path),\
            f'all paths {img_path} should exist'
        
        
        img_list = sorted(glob(os.path.join(img_path, '*.jpg')))

        # select training frames
        selected_frames=[]
        all_frames = list(range(len(img_list)))
        
        first_frame = random.sample(all_frames, 1)
        selected_frames.append(first_frame[0])
        all_frames.pop(first_frame[0])
        
        for i in range(self.num_frames-1):
            if len(all_frames)<1:
                break
            last_select = selected_frames[i]
            select_range = all_frames[max(0,last_select-7):min(last_select+7,len(all_frames)-1)]
            
            if len(select_range)< 1:
                select_range = all_frames
            selected_frame = random.sample(select_range, 1)         
            selected_frames.append(selected_frame[0])
            before_num=0
            for j in range(len(selected_frames)):
                if selected_frames[j] < selected_frame[0]:
                    before_num += 1
            all_frames.pop(selected_frame[0]-before_num)
            
        selected_frames.sort()
        
        if len(selected_frames)!=self.num_frames:
            for i in range(self.num_frames-len(selected_frames)):
                selected_frames.append(selected_frames[-1])
        
        # generate training snippets
        img_lst = []
        flow_lst = []
        mask_lst = []
        for i, frame_id in enumerate(selected_frames):
            img_path = img_list[frame_id]
            flow_path = img_path.replace('frame', 'flow')
            mask_path = img_path.replace('frame', 'mask')
            mask_path = mask_path.replace('jpg', 'png')
            
            img = self.rgb_loader(img_path)
            flow = self.rgb_loader(flow_path)
            #mask = load_image_in_PIL(mask_list[frame_id], 'P')         #mask 读取出现问题，使得读取出的图片to tensor后值都为0。此处需打开二值单通道图片
            mask = self.binary_loader(mask_path)
            #duts数据处理需修改类似问题，可直接通过self.to_tensor = tv.transforms.ToTensor()，self.to_tensor（mask）解决，

            if self.data_augmentation:
                result = dict(img=np.array(img))
                result['flow'] = np.array(flow)
                result['gt_semantic_seg'] = np.array(mask)
                result['seg_fields'] = []
                result['seg_fields'].append('gt_semantic_seg')
                result = self.augmentation(result)

                img = Image.fromarray(result['img'])
                flow = Image.fromarray(result['flow'])
                mask = Image.fromarray(result['gt_semantic_seg'])

            img = self.img_transform(img)
            flow = self.img_transform(flow)
            mask = self.gt_transform(mask)
            
            # convert formats
            img_lst.append(img)
            flow_lst.append(flow)
            mask_lst.append(mask)

        # gather all frames
        images = torch.stack(img_lst, 0)      #[L, C, H, W]，第一个一个为抽取的记忆总数
        flows = torch.stack(flow_lst, 0)    
        gts = torch.stack(mask_lst, 0)    

        return images, flows, gts, selected_frames

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
    
class val_video_data_loader(data.Dataset):
    def __init__(self, root, img_size, data_augmentation=False, mode='val'):
        self.img_size = img_size
        self.data_augmentation = data_augmentation
        self.images, self.flows, self.gts = [], [], []
        
        img_path=os.path.join(root, 'frame', mode)
        gt_path=os.path.join(root, 'mask', mode)
        images_seq =[]
        gts_seq =[]
        for seq_name in os.listdir(img_path):
            seq_img = os.path.join(img_path, seq_name)
            # seq_flow = os.path.join(img_path, seq_name)
            seq_gt = os.path.join(gt_path, seq_name)
            if os.path.isdir:
                images_seq += [seq_img]
            if os.path.isdir:
                gts_seq += [seq_gt]
        
        self.images = sorted(images_seq)
        self.gts = sorted(gts_seq)
        
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),  # bilinear
            transforms.ToTensor()])

        if self.data_augmentation:
            # self.resize = Resize(img_scale=(int(self.img_size*4), self.img_size), ratio_range=(0.5,2.0))
            # self.crop = RandomCrop(crop_size=(self.img_size, self.img_size), cat_max_ratio=0.8)
            self.flip = RandomFlip(prob=0.5)
            self.pmd = PhotoMetricDistortion()
            # self.pad = Pad(size=(self.img_size, self.img_size), pad_val=0, seg_pad_val=0)

        self.size = len(self.gts)

    def augmentation(self, result):
        # result = self.resize(result)
        # result = self.crop(result)
        result = self.flip(result)
        result = self.pmd(result)
        # result = self.pad(result)

        return result

    def __getitem__(self, index):
        img_path = self.images[index]
        # flow_path = self.flows[index]
        gt_path = self.gts[index]
        flow_path = img_path.replace('frame', 'flow')
        
        assert os.path.isdir(img_path) and os.path.isdir(gt_path) and os.path.isdir(flow_path),\
            f'all paths {img_path} should be a path'
            
        assert os.path.exists(img_path) and os.path.exists(gt_path) and os.path.exists(flow_path),\
            f'all paths {img_path} should exist'
        
        
        img_list = sorted(glob(os.path.join(img_path, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_path, '*.jpg')))
        mask_list = sorted(glob(os.path.join(gt_path, '*.png')))

        # select training frames
        all_frames = list(range(len(img_list)))
        
        # generate training snippets
        img_lst = []
        flow_lst = []
        mask_lst = []
        save_lst = []
        for i, frame_id in enumerate(all_frames):
            img = self.rgb_loader(img_list[frame_id])
            flow = self.rgb_loader(flow_list[frame_id])
            #mask = load_image_in_PIL(mask_list[frame_id], 'P')         #mask 读取出现问题，使得读取出的图片to tensor后值都为0。此处需打开二值单通道图片
            # mask = self.binary_loader(mask_list[frame_id])
            #duts数据处理需修改类似问题，可直接通过self.to_tensor = tv.transforms.ToTensor()，self.to_tensor（mask）解决，

            if self.data_augmentation:
                result = dict(img=np.array(img))
                result['flow'] = np.array(flow)
                # result['gt_semantic_seg'] = np.array(mask)
                result['seg_fields'] = []
                result['seg_fields'].append('gt_semantic_seg')
                result = self.augmentation(result)

                img = Image.fromarray(result['img'])
                flow = Image.fromarray(result['flow'])
                # mask = Image.fromarray(result['gt_semantic_seg'])

            img = self.img_transform(img)
            flow = self.img_transform(flow)
            # mask = self.gt_transform(mask)
            
            # convert formats
            img_lst.append(img)
            flow_lst.append(flow)
            # mask_lst.append(mask)

        # gather all frames
        images = torch.stack(img_lst, 0)      #[L, C, H, W]，第一个一个为抽取的记忆总数
        flows = torch.stack(flow_lst, 0)    
        # gts = torch.stack(mask_lst, 0)    

        return images, flows, mask_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root, use_flow):
        self.use_flow = use_flow
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        lst = []
        for name in lst_label:
            if name in lst_pred:
                lst.append(name)
        self.image_path = self.get_paths(lst, img_root)
        self.label_path = self.get_paths(lst, label_root)
        self.key_list = list(self.image_path.keys())

        self.check_path(self.image_path, self.label_path)
        self.trans = transforms.Compose([transforms.ToTensor()])


    def check_path(self, image_path_dict, label_path_dict):
        assert image_path_dict.keys() == label_path_dict.keys(), 'gt, pred must have the same videos'
        for k in image_path_dict.keys():
            assert len(image_path_dict[k]) == len(label_path_dict[k]), f'{k} have different frames'

    def get_paths(self, lst, root):
        v_lst = list(map(lambda x: os.path.join(root, x), lst))

        f_lst = {}
        for v in v_lst:
            v_name = v.split('/')[-1]
            if 'MATNet' in root:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]  # 光流方法忽略第一帧和最后一帧
            else:
                if not self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:]
                elif self.use_flow:
                    f_lst[v_name] = sorted([os.path.join(v, f) for f in os.listdir(v)])[1:-1]  # 光流方法忽略第一帧和最后一帧
        return f_lst

    def read_picts(self, v_name):
        pred_names = self.image_path[v_name]
        pred_list = []
        for pred_n in pred_names:
            pred_list.append(self.trans(Image.open(pred_n).convert('L')))

        gt_names = self.label_path[v_name]
        gt_list = []
        for gt_n in gt_names:
            gt_list.append(self.trans(Image.open(gt_n).convert('L')))

        for i in range(len(gt_list)):
            if gt_list[i].shape!= pred_list[i].shape:
                print(f"{gt_names[i]} gt.shape {gt_list[i].shape} != pred.shape {pred_list[i].shape}, remove {gt_names[i]}")
                del gt_list[i]
                del pred_list[i]
        # for gt, pred in zip(gt_list, pred_list):
        #     assert gt.shape == pred.shape, 'gt.shape!=pred.shape'
        
        gt_list = torch.cat(gt_list,dim=0)
        pred_list = torch.cat(pred_list,dim=0)
        return pred_list, gt_list

    def __getitem__(self, item):
        v_name = self.key_list[item]
        preds, gts = self.read_picts(v_name)
        
        return v_name, preds, gts

    def __len__(self):
        return len(self.image_path)
