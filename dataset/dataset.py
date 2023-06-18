# Datasets of 3D reconstruction from vision + touch
# Yiming Dou (yimingdou@cs.stanford.edu)
# May 2022

import os
import os.path as osp
import json
from tqdm import tqdm
import random

import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

class reconstruction_dataset(Dataset):
    def __init__(self, args, set_type='train'):
        self.args = args
        self.use_vision='vision' in args.modality_list
        self.use_touch='touch' in args.modality_list
        self.use_audio='audio' in args.modality_list
        self.set_type = set_type  # 'train' or 'val' or 'test'
        self.global_gt_points_location = self.args.global_gt_points_location
        self.local_gt_points_location = self.args.local_gt_points_location
        self.visual_images_location = self.args.visual_images_location
        self.tactile_images_location = self.args.tactile_images_location
        self.tactile_depth_location = self.args.tactile_depth_location
        self.camera_info_location = self.args.camera_info_location
        self.audio_spectrogram_location = self.args.audio_spectrogram_location
        
        # sample size of global GT point clouds
        self.global_sample_num = self.args.global_sample_num
        # sample size of local GT point clouds
        self.local_sample_num = self.args.local_sample_num
        
        # if True, normalize each point cloud into unit globe
        self.normalize = self.args.normalize
        
        self.preprocess = {
            'vision': T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]),
            'touch': T.Compose([
                # T.Resize((120, 160)),
                T.CenterCrop(160),
                T.Resize((256, 256)),
                T.ToTensor(),
            ]),
            'audio': T.Compose([
                T.ToTensor(),
            ])
        }
        with open(self.args.split_location) as f:
            self.cand = json.load(f)[self.set_type] # [[obj, view, touch_list, audio_list],]
            
    def __len__(self):
        return len(self.cand)
    
    # load the #view visual RGB image of obj
    def load_visual_image(self, obj, view):
        visual_image = Image.open(
            osp.join(self.visual_images_location, obj, f'{view}.png')
        ).convert('RGB')
        visual_image = self.preprocess['vision'](visual_image)
        return torch.FloatTensor(visual_image)
    
    # load the #touch tactile readings (RGB + depth) of obj
    def load_tactile_readings(self, obj, touch):
        tactile_image = Image.open(
            osp.join(self.tactile_images_location, obj, f'{touch}.png')
        ).convert('RGB')
        tactile_image = self.preprocess['touch'](tactile_image)
        tactile_depth = np.load(
            osp.join(self.tactile_depth_location, obj, f'{touch}.npy')
        )
        return torch.FloatTensor(tactile_image), torch.FloatTensor(tactile_depth)

    # load the #audio spetrogram of obj
    def load_spectrogram(self, obj, audio):
        audio_path = osp.join(self.audio_spectrogram_location,obj,'{}.npy'.format(audio))
        spectrogram = np.load(audio_path)
        return torch.FloatTensor(spectrogram)
    
    # load the surface point cloud of obj
    def load_global_gt_points(self, obj):
        global_gt_points = np.load(
            osp.join(self.global_gt_points_location, f'{obj}.npy'))
        global_gt_points = global_gt_points[np.random.choice(global_gt_points.shape[0],
                                                             self.global_sample_num,
                                                             replace=False)]
        global_gt_points = torch.FloatTensor(global_gt_points)
        return global_gt_points
    
    # load the #touch local point cloud of obj
    def load_local_gt_points(self, obj, touch):
        local_gt_points = np.load(
            osp.join(self.local_gt_points_location, obj, f'{touch}.npy'))
        local_gt_points = local_gt_points[np.random.choice(local_gt_points.shape[0],
                                                            min(self.local_sample_num,local_gt_points.shape[0]),
                                                            replace=False)]
        local_gt_points = torch.FloatTensor(local_gt_points)
        return local_gt_points
    
    # load the #touch camera information (position + rotation) of obj
    def load_camera_info(self, obj, touch):
        data = np.load(osp.join(self.camera_info_location, obj, f'{touch}.npy'),allow_pickle=True).item()
        camera_info = {}
        rot = data['cam_rot']
        pos = data['cam_pos']
        rot_M = R.from_euler('xyz', rot, degrees=False).as_matrix()
        rot_q = R.from_matrix(rot_M).as_quat()
        camera_info['rot'] = torch.FloatTensor(rot_q)
        camera_info['rot_M'] = torch.FloatTensor(rot_M)
        camera_info['pos'] = torch.FloatTensor(pos)
        return camera_info
    
    # normalize the point cloud into the unit globe
    def normalize_points(self, batch):
        if self.normalize:
            mean = torch.mean(batch['global_gt_points'], dim=1).unsqueeze(1)
            batch['global_gt_points'] -= mean
            scale = torch.max(torch.sqrt(torch.sum(batch['global_gt_points'] ** 2, dim=2)),dim=1).values.unsqueeze(1).unsqueeze(1)
            batch['global_gt_points'] /= scale
            batch['scale'] = scale
            if 'local_gt_points' in batch:
                batch['local_gt_points'] = (batch['local_gt_points'] - mean) / scale
            if 'camera_info' in batch:
                batch['camera_info']['pos'] = (batch['camera_info']['pos'] - mean) / scale
        else:
            # batch['scale'] = torch.ones_like(batch['global_gt_points'])
            batch['scale'] = torch.ones((batch['global_gt_points'].shape[0],1,1))
        return batch
    
    def __getitem__(self, index):
        obj, view, touch_list, audio_list = self.cand[index]
        if self.set_type == 'train':
            index_list = np.arange(len(touch_list))
            np.random.shuffle(index_list)
            touch_list = np.array(touch_list)[index_list].tolist()
            audio_list = np.array(audio_list)[index_list].tolist()
        data = {}
        data['names'] = (obj, touch_list[:self.args.touch_num], audio_list[:self.args.audio_num])
        data['global_gt_points'] = self.load_global_gt_points(obj) # (self.global_sample_num, 3)
        if self.use_vision:
            data['visual_image'] = self.load_visual_image(obj, view) # (3, 256, 256)
        if self.use_touch:
            data['local_gt_points'] = []
            data['tactile_image'] = []
            data['tactile_depth'] = []
            data['camera_info'] = {
                'rot':[], 'rot_M':[], 'pos':[]
            }
            assert len(touch_list) >= self.args.touch_num
            if self.set_type == 'train':
                random.shuffle(touch_list)
            for touch in touch_list[:self.args.touch_num]:
                # local_gt_points = self.load_local_gt_points(obj, touch)
                camera_info = self.load_camera_info(obj, touch)
                tactile_image, tactile_depth = self.load_tactile_readings(obj, touch) # (3, 120, 160), (120, 160)
                # data['local_gt_points'].append(local_gt_points)
                data['camera_info']['rot'].append(camera_info['rot'])
                data['camera_info']['rot_M'].append(camera_info['rot_M'])
                data['camera_info']['pos'].append(camera_info['pos'])
                data['tactile_image'].append(tactile_image)
                # data['tactile_depth'].append(tactile_depth)
            # data['local_gt_points'] = torch.cat(data['local_gt_points']) # (touch_num * self.local_sample_num, 3)
            data['tactile_image'] = torch.stack(data['tactile_image']) # (touch_num, 3, 120, 160)
            # data['tactile_depth'] = torch.stack(data['tactile_depth']) # (touch_num, 120, 160)
            data['camera_info']['rot'] = torch.stack(data['camera_info']['rot'])
            data['camera_info']['rot_M'] = torch.stack(data['camera_info']['rot_M'])
            data['camera_info']['pos'] = torch.stack(data['camera_info']['pos'])
        if self.use_audio:
            if not 'camera_info' in data:
                data['camera_info'] = {
                    'rot':[], 'rot_M':[], 'pos':[]
                }
                for audio in audio_list[:self.args.audio_num]:
                    camera_info = self.load_camera_info(obj, audio)
                    data['camera_info']['rot'].append(camera_info['rot'])
                    data['camera_info']['rot_M'].append(camera_info['rot_M'])
                    data['camera_info']['pos'].append(camera_info['pos'])
                data['camera_info']['rot'] = torch.stack(data['camera_info']['rot'])
                data['camera_info']['rot_M'] = torch.stack(data['camera_info']['rot_M'])
                data['camera_info']['pos'] = torch.stack(data['camera_info']['pos'])
            data['spectrogram'] = [
                self.load_spectrogram(obj, audio) for audio in audio_list[:self.args.audio_num]]
            data['spectrogram'] = torch.stack(data['spectrogram'])  # (audio_num, 257, 301)
            
        
        return data
    
    def collate(self, data):
        batch = {}
        batch['names'] = [item['names'] for item in data]
        batch['global_gt_points'] = torch.cat([item['global_gt_points'].unsqueeze(0) for item in data]) # (bs, self.global_sample_num, 3)
        if self.use_vision:
            batch['visual_image'] = torch.cat([item['visual_image'].unsqueeze(0) for item in data]) # (bs, 3, 256, 256)
        if self.use_touch:
            # batch['local_gt_points'] = torch.cat([item['local_gt_points'].unsqueeze(0) for item in data]) # (bs, touch_num * self.local_sample_num, 3)
            batch['tactile_image'] = torch.cat([item['tactile_image'].unsqueeze(0) for item in data]) # (bs, touch_num, 3, 120, 160)
            # batch['tactile_depth'] = torch.cat([item['tactile_depth'].unsqueeze(0) for item in data]) # (bs, touch_num, 120, 160)
            batch['camera_info']={}
            batch['camera_info']['rot'] = torch.stack([item['camera_info']['rot'] for item in data])
            batch['camera_info']['rot_M'] = torch.stack([item['camera_info']['rot_M'] for item in data])
            batch['camera_info']['pos'] = torch.stack([item['camera_info']['pos'] for item in data])
        if self.use_audio:
            batch['spectrogram'] = torch.cat([item['spectrogram'].unsqueeze(0) for item in data]) # (bs, audio_num, 257, 301)
            if not 'camera_info' in batch:
                batch['camera_info']={}
                batch['camera_info']['rot'] = torch.stack([item['camera_info']['rot'] for item in data])
                batch['camera_info']['rot_M'] = torch.stack([item['camera_info']['rot_M'] for item in data])
                batch['camera_info']['pos'] = torch.stack([item['camera_info']['pos'] for item in data])
        
        batch = self.normalize_points(batch)
        
        return batch