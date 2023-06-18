import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


class Encoder(nn.Module):
    def __init__(self, touch_num):
        super(Encoder, self).__init__()

        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        # second shared mlp
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n = x.size()[2]

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 128, N)
        f = self.bn2(self.conv2(x))                   # (B, 256, N)

        # point-wise maxpool
        g = torch.max(f, dim=2, keepdim=True)[0]      # (B, 256, 1)

        # expand and concat
        x = torch.cat([g.repeat(1, 1, n), f], dim=1)  # (B, 512, N)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 512, N)
        x = self.bn4(self.conv4(x))                   # (B, 1024, N)

        # point-wise maxpool
        v = torch.max(x, dim=-1)[0]                   # (B, 1024)

        return v


class Decoder(nn.Module):
    def __init__(self, num_coarse=4*1024):
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
    def __init__(self, touch_num):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(touch_num)
        self.decoder = Decoder()

    def forward(self, batch, calc_loss = False):
        output={}
        x = batch['local_gt_points'].permute(0, 2, 1).cuda()
        h = self.encoder(x)
        global_pred_points = self.decoder(h).permute(0, 2, 1)
        
        output['h'] = h
        output['global_pred_points'] = global_pred_points
        
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_gt_points, global_pred_points)[0]
        return output
