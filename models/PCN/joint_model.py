import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import touch_model
import vision_model
import audio_model
from pytorch3d.loss import chamfer_distance

class AutoEncoder(nn.Module):
    def __init__(self, args, cfg, use_vision=False, use_touch=False, use_audio=False):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_vision = use_vision
        self.use_touch = use_touch
        self.use_audio = use_audio
        
        if self.use_vision:
            self.vision_encoder = vision_model.AutoEncoder()
        if self.use_touch:
            self.tactile_encoder = touch_model.AutoEncoder(self.args.touch_num)
        if self.use_audio:
            self.audio_encoder = audio_model.AutoEncoder(self.args.audio_num)
        
        if 'pretrain' in self.cfg.keys():
            if 'vision_backbone' in self.cfg.pretrain.keys() and self.use_vision:
                print(f"loading vision backbone from {self.cfg.pretrain.vision_backbone}")
                vision_backbone_state_dict = torch.load(self.cfg.pretrain.vision_backbone,map_location='cpu')
                self.vision_encoder.encoder.backbone.load_state_dict(vision_backbone_state_dict)
            if 'audio_backbone' in self.cfg.pretrain.keys() and self.use_audio:
                print(f"loading audio backbone from {self.cfg.pretrain.audio_backbone}")
                audio_backbone_state_dict = torch.load(self.cfg.pretrain.audio_backbone,map_location='cpu')
                self.audio_encoder.encoder.backbone.load_state_dict(audio_backbone_state_dict)
            
        self.linear1 = nn.Linear(1024*4*(self.use_vision + self.use_touch + self.use_audio), 1024*4)
        self.linear2 = nn.Linear(1024*4, 1024*4)
        # self.linear = nn.Sequential(
        #     nn.Linear(1024*4*(self.use_vision + self.use_touch + self.use_audio), 1024*4),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024*4, 1024*4),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024*4, 1024*4),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024*4, 1024*4),
        # )
        
    def forward(self, batch, calc_loss = False):
        output={}
        h = []
        if self.use_vision:
            h.append(self.vision_encoder(batch)['global_pred_points'].permute(0, 2, 1))  # (B, 3, 1024*4)
        if self.use_touch:
            h.append(self.tactile_encoder(batch)['global_pred_points'].permute(0, 2, 1))  # (B, 3, 1024*4)
        if self.use_audio:
            h.append(self.audio_encoder(batch)['global_pred_points'].permute(0, 2, 1))  # (B, 3, 1024*4)
            
        h = torch.cat(h, dim=2)  # (B, 3, 4096*modality_num)
        
        h = F.relu(self.linear1(h))  # (B, 3, 1024*4)
        global_pred_points = self.linear2(h).permute(0, 2, 1)  # (B, 3, 1024*4)
        
        # global_pred_points = self.linear(h).permute(0, 2, 1)
        output['global_pred_points'] = global_pred_points
        
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_gt_points, global_pred_points)[0]
        
        return output
