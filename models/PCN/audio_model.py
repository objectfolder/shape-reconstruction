import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch3d.loss import chamfer_distance

class AudioBackbone(nn.Module):
    def __init__(self, pretrained=False,  num_class=100):
        super(AudioBackbone, self).__init__()
        original_resnet = models.resnet18(pretrained)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.apply(self.weights_init)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) 
    
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        return x

class Encoder(nn.Module):
    def __init__(self, audio_num):
        super(Encoder, self).__init__()

        self.backbone = AudioBackbone(pretrained=True)
        self.linear = nn.Linear(512*audio_num+7*audio_num, 1024)

    def forward(self, spectrogram, position_info):
        h = [self.backbone(spectrogram[:, i].unsqueeze(1)) for i in range(spectrogram.shape[1])]
        h = torch.flatten(torch.cat(h, 1), 1)
        x = torch.cat([h, position_info],dim=-1)
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
    def __init__(self, audio_num):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(audio_num)
        self.decoder = Decoder()
    
    def forward(self, batch, calc_loss = False):
        output={}
        spectrogram = batch['spectrogram'][:,:,:,:batch['spectrogram'].shape[2]].cuda() # (bs, audio_num, 257, 257)
        position_info = torch.stack([torch.stack([torch.cat([j['rot'], j['pos']]) for j in item])
                                   for item in batch['camera_info']]).cuda() # (bs, impact_num, 7)
        position_info = torch.flatten(position_info, 1) # (bs, impact_num*7)
        
        h = self.encoder(spectrogram, position_info) # (bs, 1024)

        global_pred_points = self.decoder(h).permute(0, 2, 1)
        
        output['h'] = h
        output['global_pred_points'] = global_pred_points
        
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_gt_points, global_pred_points)[0]
        
        return output