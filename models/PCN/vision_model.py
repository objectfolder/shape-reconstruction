import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.loss import chamfer_distance

class VisionBackbone(nn.Module):
    def __init__(self, pretrained=False, num_class=100):
        super(VisionBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers) 
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.backbone = VisionBackbone(pretrained=True)
        self.linear = nn.Linear(512, 1024)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_coarse=1024*4):
        super(Decoder, self).__init__()

        self.num_coarse = num_coarse
        
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x  # (B, 1024)

        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 1024)

        return y_coarse


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, batch, calc_loss = False):
        output={}
        x = batch['visual_image'].cuda()
        h = self.encoder(x)
        global_pred_points = self.decoder(h).permute(0, 2, 1)
        
        output['h'] = h
        output['global_pred_points'] = global_pred_points
        
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_gt_points, global_pred_points)[0]
        
        return output