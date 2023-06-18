import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import chamfer_distance
from einops import rearrange,repeat

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNetBackbone, self).__init__()
        original_resnet = resnet18(pretrained)
        layers = list(original_resnet.children())[0:-1]
        self.feature_extraction = nn.Sequential(*layers) 
    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1)
        return x

class AudioResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioResNetBackbone, self).__init__()
        original_resnet = resnet18(pretrained)
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

class VisualEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(VisualEncoder, self).__init__()
        self.backbone = ResNetBackbone()
        self.linear = nn.Linear(512, feature_dim)
    
    def forward(self, batch):
        output={}
        visual_image = batch['visual_image'].cuda()
        visual_feature = self.backbone(visual_image)
        visual_feature = self.linear(visual_feature)
        return visual_feature # (bs, feature_dim)

class TactileEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(TactileEncoder, self).__init__()
        self.backbone = ResNetBackbone()
        self.linear = nn.Linear(512, feature_dim)
        self.pos_encoder = nn.Linear(7, feature_dim)
        self.recon_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                    nhead=8, dim_feedforward=512,
                                                    activation='gelu',
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, batch):
        output={}
        tactile_image = batch['tactile_image'].cuda()
        batch_size = tactile_image.size(0)
        tactile_feature = self.backbone(
            rearrange(tactile_image, 'b1 b2 c h w -> (b1 b2) c h w'))
        tactile_feature = self.linear(tactile_feature)
        tactile_feature = rearrange(tactile_feature, '(b1 b2) c -> b1 b2 c', b1=batch_size)
        camera_info = torch.cat([batch['camera_info']['rot'],batch['camera_info']['pos']],dim=2).cuda()
        pos_feature = self.pos_encoder(camera_info)
        tactile_feature += pos_feature # (bs, seq, feature_dim)
        recon_tokens = repeat(self.recon_token, '() n d -> b n d', b=batch_size) # (bs, 1, feature_dim)
        transformer_input = torch.cat([recon_tokens, tactile_feature], dim=1) # (bs, seq+1, feature_dim)
        transformer_output = self.transformer_encoder(transformer_input) # (bs, seq+1, feature_dim)
        tactile_feature = transformer_output[:, 0, :] # (bs, feature_dim)
        
        return tactile_feature
    
class AudioEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(AudioEncoder, self).__init__()
        self.backbone = AudioResNetBackbone()
        self.linear = nn.Linear(512, feature_dim)
        self.pos_encoder = nn.Linear(7, feature_dim)
        self.recon_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                                    nhead=8, dim_feedforward=512,
                                                    activation='gelu',
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, batch):
        output={}
        spectrogram = batch['spectrogram'].cuda()
        spectrogram = spectrogram[:,:,:,:spectrogram.shape[2]].unsqueeze(dim=2)
        batch_size = spectrogram.size(0)
        audio_feature = self.backbone(
            rearrange(spectrogram, 'b1 b2 c h w -> (b1 b2) c h w'))
        audio_feature = self.linear(audio_feature)
        audio_feature = rearrange(audio_feature, '(b1 b2) c -> b1 b2 c', b1=batch_size)
        
        camera_info = torch.cat([batch['camera_info']['rot'],batch['camera_info']['pos']],dim=2).cuda()
        pos_feature = self.pos_encoder(camera_info)
        audio_feature += pos_feature # (bs, seq, feature_dim)
        recon_tokens = repeat(self.recon_token, '() n d -> b n d', b=batch_size) # (bs, 1, feature_dim)
        transformer_input = torch.cat([recon_tokens, audio_feature], dim=1) # (bs, seq+1, feature_dim)
        transformer_output = self.transformer_encoder(transformer_input) # (bs, seq+1, feature_dim)
        audio_feature = transformer_output[:, 0, :] # (bs, feature_dim)
        
        return audio_feature
    
# Transformer-based 3D reconstruction model
class ReconstructionTransformer(nn.Module):
    def __init__(self, args, cfg, feature_dim=512, num_points=1024,
                use_vision=False, use_touch=False, use_audio=False):
        super(ReconstructionTransformer, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_vision = use_vision
        self.use_touch = use_touch
        self.use_audio = use_audio
        
        decoder_input_dim = 0
        if self.use_vision:
            self.visual_encoder = VisualEncoder(feature_dim)
            decoder_input_dim+=feature_dim
        if self.use_touch:
            self.tactile_encoder = TactileEncoder(feature_dim)
            if self.use_vision:
                self.tactile_head = nn.Linear(feature_dim, 32)
                decoder_input_dim += 32
            else:
                decoder_input_dim+=feature_dim
        if self.use_audio:
            self.audio_encoder = AudioEncoder(feature_dim)
            if self.use_vision or self.use_touch:
                self.audio_head = nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Linear(feature_dim, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 8),
                )
                decoder_input_dim += 8
            else:
                decoder_input_dim += feature_dim
            
        if 'pretrain' in self.cfg.keys():
            if 'visual_backbone' in self.cfg.pretrain.keys() and self.use_vision:
                print(f"loading visual backbone from {self.cfg.pretrain.visual_backbone}")
                visual_backbone_state_dict = torch.load(self.cfg.pretrain.visual_backbone, map_location='cpu')
                try:
                    self.visual_encoder.backbone.load_state_dict(visual_backbone_state_dict)
                except:
                    visual_backbone_state_dict={k: v for 
                                                k, v in visual_backbone_state_dict.items() 
                                                if 'visual_encoder' in k}
                    self.load_state_dict(visual_backbone_state_dict, strict=False)
            if 'tactile_backbone' in self.cfg.pretrain.keys() and self.use_touch:
                print(f"loading tactile backbone from {self.cfg.pretrain.tactile_backbone}")
                tactile_backbone_state_dict = torch.load(self.cfg.pretrain.tactile_backbone, map_location='cpu')
                try:
                    self.tactile_encoder.backbone.load_state_dict(tactile_backbone_state_dict)
                except:
                    tactile_backbone_state_dict={k: v for 
                                                k, v in tactile_backbone_state_dict.items() 
                                                if 'tactile_encoder' in k}
                    self.load_state_dict(tactile_backbone_state_dict, strict=False)
            if 'audio_backbone' in self.cfg.pretrain.keys() and self.use_audio:
                print(f"loading audio backbone from {self.cfg.pretrain.audio_backbone}")
                audio_backbone_state_dict = torch.load(self.cfg.pretrain.audio_backbone, map_location='cpu')
                try:
                    self.audio_encoder.backbone.load_state_dict(audio_backbone_state_dict)
                except:
                    audio_backbone_state_dict={k: v for 
                                                k, v in audio_backbone_state_dict.items() 
                                                if 'audio_encoder' in k}
                    self.load_state_dict(audio_backbone_state_dict, strict=False)
        
        self.point_decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, feature_dim*3),
            nn.BatchNorm1d(feature_dim*3),
            nn.LeakyReLU(),
            nn.Linear(feature_dim*3, num_points*3),
        )
        self.num_points = num_points
        for param in self.named_parameters():
            if 'visual_encoder' in param[0] \
                or 'tactile_encoder' in param[0] \
                    or 'audio_encoder' in param[0]:
                param[1].requires_grad = False

    def forward(self, batch, calc_loss=True):
        output={}
        batch_size = batch['global_gt_points'].size(0)
        decoder_input = []
        if self.use_vision:
            visual_feature = self.visual_encoder(batch)
            decoder_input.append(visual_feature)
        if self.use_touch:
            tactile_feature = self.tactile_encoder(batch)
            if self.use_vision:
                tactile_feature = self.tactile_head(tactile_feature)
            decoder_input.append(tactile_feature)
        if self.use_audio:
            audio_feature = self.audio_encoder(batch)
            if self.use_vision or self.use_touch:
                audio_feature = self.audio_head(audio_feature)
            decoder_input.append(audio_feature)
        
        decoder_input = torch.cat(decoder_input, dim=1)
        global_pred_points = self.point_decoder(decoder_input).view(batch_size, self.num_points, 3)
        output['global_pred_points'] = global_pred_points
        
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_gt_points, global_pred_points)[0]
        return output