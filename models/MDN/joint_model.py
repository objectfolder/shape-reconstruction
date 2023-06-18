import os, sys
import os.path as osp
sys.path.append(osp.dirname(osp.abspath(__file__)))
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
import mdn_utils
from ipdb import set_trace

# CNN layer definition
def CNN_layer(f_in, f_out, k, stride=1, max_pool = False):
    layers = []
    layers.append(nn.Conv2d(int(f_in), int(f_out),
                  kernel_size=k, padding=1, stride=stride))
    if max_pool:
        layers.append(nn.MaxPool2d(2, 2))
    layers.append(nn.BatchNorm2d(int(f_out)))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

# network for making image features for vertex feature vectors
class Image_Encoder(nn.Module):
    def __init__(self):
        super(Image_Encoder, self).__init__()
        layers = []
        next_size = 16
        self.num_img_blocks = 6
        self.num_img_layers = 3
        self.size_img_ker = 5

        cur_size = 3
        for i in range(self.num_img_blocks):
            layers.append(CNN_layer(cur_size, next_size, self.size_img_ker, stride=2))
            cur_size = next_size
            next_size = next_size * 2
            for j in range(self.num_img_layers - 1):
                layers.append(CNN_layer(cur_size, cur_size, self.size_img_ker))

        self.layers = nn.ModuleList(layers)
        f = 221.7025
        RT = np.array([[-0.0000, -1.0000, 0.0000, -0.0000],
                       [-0.7071, 0.0000, -0.7071, 0.4243],
                       [0.7071, 0.0000, -0.7071, 1.1314]])
        K = np.array([[f, 0, 128.], [0, f, 128.], [0, 0, 1]])
        self.matrix = torch.FloatTensor(K.dot(RT)).cuda()

    def pooling(self, blocks, verts_pos, debug=False):
        # convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension
        ext_verts_pos = torch.cat(
            (verts_pos, torch.FloatTensor(np.ones([verts_pos.shape[0], verts_pos.shape[1], 1])).cuda()), dim=-1)
        ext_verts_pos = torch.matmul(ext_verts_pos, self.matrix.permute(1, 0))
        xs = ext_verts_pos[:, :, 1] / ext_verts_pos[:, :, 2] / 256.
        ys = ext_verts_pos[:, :, 0] / ext_verts_pos[:, :, 2] / 256.

        full_features = None
        batch_size = verts_pos.shape[0]

        # check camera project covers the image
        if debug:
            dim = 256
            xs = (torch.clamp(xs * dim, 0, dim -
                  1).data.cpu().numpy()).astype(np.uint8)
            ys = (torch.clamp(ys * dim, 0, dim -
                  1).data.cpu().numpy()).astype(np.uint8)
            for ex in range(blocks.shape[0]):
                img = blocks[ex].permute(1, 2, 0).data.cpu().numpy()[:, :, :3]
                for x, y in zip(xs[ex], ys[ex]):
                    img[x, y, 0] = 1
                    img[x, y, 1] = 0
                    img[x, y, 2] = 0

                from PIL import Image
                Image.fromarray((img * 255).astype(np.uint8)
                                ).save('results/temp.png')
                print('saved')
                input()

        for block in blocks:
            # scale projected vertex points to dimension of current feature map
            dim = block.shape[-1]
            cur_xs = torch.clamp(xs * dim, 0, dim - 1)
            cur_ys = torch.clamp(ys * dim, 0, dim - 1)

            # https://en.wikipedia.org/wiki/Bilinear_interpolation
            x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(
                cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)
            A = x2s - cur_xs
            B = cur_xs - x1s
            G = y2s - cur_ys
            H = cur_ys - y1s

            x1s = x1s.type(torch.cuda.LongTensor)
            y1s = y1s.type(torch.cuda.LongTensor)
            x2s = x2s.type(torch.cuda.LongTensor)
            y2s = y2s.type(torch.cuda.LongTensor)

            # flatten batch of feature maps to make vectorization easier
            flat_block = block.permute(
                1, 0, 2, 3).contiguous().view(block.shape[1], -1)
            block_idx = torch.arange(0, verts_pos.shape[0]).cuda(
            ).unsqueeze(-1).expand(batch_size, verts_pos.shape[1])
            block_idx = block_idx * dim * dim
            # set_trace()

            selection = (block_idx + (x1s * dim) + y1s).view(-1)
            C = torch.index_select(flat_block, 1, selection)
            C = C.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x1s * dim) + y2s).view(-1)
            D = torch.index_select(flat_block, 1, selection)
            D = D.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x2s * dim) + y1s).view(-1)
            E = torch.index_select(flat_block, 1, selection)
            E = E.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x2s * dim) + y2s).view(-1)
            F = torch.index_select(flat_block, 1, selection)
            F = F.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)

            section1 = A.unsqueeze(1) * C * G.unsqueeze(1)
            section2 = H.unsqueeze(1) * D * A.unsqueeze(1)
            section3 = G.unsqueeze(1) * E * B.unsqueeze(1)
            section4 = B.unsqueeze(1) * F * H.unsqueeze(1)

            features = (section1 + section2 + section3 + section4)
            features = features.permute(0, 2, 1)

            if full_features is None:
                full_features = features
            else:
                full_features = torch.cat((full_features, features), dim=2)

        return full_features

    def forward(self, cur_vertices, visual_image):
        x = visual_image
        # set_trace()
        features = []
        layer_selections = [len(self.layers) - 1 - (i+1)* self.num_img_layers for i in range(3)]
        for e, layer in enumerate(self.layers):
            if x.shape[-1] < self.size_img_ker:
                break
            x = layer(x)
            # collect feature maps
            if e in layer_selections:
                features.append(x)
        features.append(x)
        # get vertex features from selected feature maps
        vert_image_features = self.pooling(features, cur_vertices)
        return vert_image_features

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

class Audio_Encoder(nn.Module):
    def __init__(self, cfg, audio_num = 8):
        super(Audio_Encoder, self).__init__()
        self.cfg = cfg
        cur_size = audio_num
        next_size = 16
        self.num_blocks = 3
        self.num_layers = 2
        layers = []
        for i in range(self.num_blocks):
            layers.append(CNN_layer(cur_size, next_size, 5, stride=2, max_pool = i < 2))
            cur_size = next_size
            next_size = next_size * 2
            for j in range(self.num_layers - 1):
                layers.append(CNN_layer(cur_size, cur_size, 5))
        
        self.backbone = AudioBackbone(pretrained=True)
        if 'pretrain' in self.cfg.keys():
            if 'audio_backbone' in self.cfg.pretrain.keys():
                print(f"loading audio backbone from {self.cfg.pretrain.audio_backbone}")
                audio_backbone_state_dict = torch.load(self.cfg.pretrain.audio_backbone,map_location='cpu')
                self.backbone.load_state_dict(audio_backbone_state_dict)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

        self.layers = nn.ModuleList(layers)
        f = 221.7025
        RT = np.array([[-0.0000, -1.0000, 0.0000, -0.0000],
                       [-0.7071, 0.0000, -0.7071, 0.4243],
                       [0.7071, 0.0000, -0.7071, 1.1314]])
        K = np.array([[f, 0, 128.], [0, f, 128.], [0, 0, 1]])
        self.matrix = torch.FloatTensor(K.dot(RT)).cuda()
        
    def pooling(self, blocks, verts_pos, debug=False):
        # convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension
        ext_verts_pos = torch.cat(
            (verts_pos, torch.FloatTensor(np.ones([verts_pos.shape[0], verts_pos.shape[1], 1])).cuda()), dim=-1)
        ext_verts_pos = torch.matmul(ext_verts_pos, self.matrix.permute(1, 0))
        xs = ext_verts_pos[:, :, 1] / ext_verts_pos[:, :, 2] / 256.
        ys = ext_verts_pos[:, :, 0] / ext_verts_pos[:, :, 2] / 256.

        full_features = None
        batch_size = verts_pos.shape[0]

        # check camera project covers the image
        if debug:
            dim = 256
            xs = (torch.clamp(xs * dim, 0, dim -
                  1).data.cpu().numpy()).astype(np.uint8)
            ys = (torch.clamp(ys * dim, 0, dim -
                  1).data.cpu().numpy()).astype(np.uint8)
            for ex in range(blocks.shape[0]):
                img = blocks[ex].permute(1, 2, 0).data.cpu().numpy()[:, :, :3]
                for x, y in zip(xs[ex], ys[ex]):
                    img[x, y, 0] = 1
                    img[x, y, 1] = 0
                    img[x, y, 2] = 0

                from PIL import Image
                Image.fromarray((img * 255).astype(np.uint8)
                                ).save('results/temp.png')
                print('saved')
                input()

        for block in blocks:
            # scale projected vertex points to dimension of current feature map
            dim = block.shape[-1]
            cur_xs = torch.clamp(xs * dim, 0, dim - 1)
            cur_ys = torch.clamp(ys * dim, 0, dim - 1)

            # https://en.wikipedia.org/wiki/Bilinear_interpolation
            x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(
                cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)
            A = x2s - cur_xs
            B = cur_xs - x1s
            G = y2s - cur_ys
            H = cur_ys - y1s

            x1s = x1s.type(torch.cuda.LongTensor)
            y1s = y1s.type(torch.cuda.LongTensor)
            x2s = x2s.type(torch.cuda.LongTensor)
            y2s = y2s.type(torch.cuda.LongTensor)

            # flatten batch of feature maps to make vectorization easier
            flat_block = block.permute(
                1, 0, 2, 3).contiguous().view(block.shape[1], -1)
            block_idx = torch.arange(0, verts_pos.shape[0]).cuda(
            ).unsqueeze(-1).expand(batch_size, verts_pos.shape[1])
            block_idx = block_idx * dim * dim
            # set_trace()

            selection = (block_idx + (x1s * dim) + y1s).view(-1)
            C = torch.index_select(flat_block, 1, selection)
            C = C.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x1s * dim) + y2s).view(-1)
            D = torch.index_select(flat_block, 1, selection)
            D = D.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x2s * dim) + y1s).view(-1)
            E = torch.index_select(flat_block, 1, selection)
            E = E.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)
            selection = (block_idx + (x2s * dim) + y2s).view(-1)
            F = torch.index_select(flat_block, 1, selection)
            F = F.view(-1, batch_size, verts_pos.shape[1]).permute(1, 0, 2)

            section1 = A.unsqueeze(1) * C * G.unsqueeze(1)
            section2 = H.unsqueeze(1) * D * A.unsqueeze(1)
            section3 = G.unsqueeze(1) * E * B.unsqueeze(1)
            section4 = B.unsqueeze(1) * F * H.unsqueeze(1)

            features = (section1 + section2 + section3 + section4)
            features = features.permute(0, 2, 1)

            if full_features is None:
                full_features = features
            else:
                full_features = torch.cat((full_features, features), dim=2)

        return full_features

    def forward(self, cur_vertices, spectrogram):
        x = spectrogram
        # features = [self.encoder(x[:,i].unsqueeze(1)) for i in range(x.shape[1])]
        features = [self.backbone(x[:,i].unsqueeze(1)) for i in range(x.shape[1])]
        features = [self.bottleneck(i).unsqueeze(2).unsqueeze(2) for i in features]
        # layer_selections = [len(self.layers) - 1 - (i+1)* self.num_layers for i in range(1, 3)]
        # for e, layer in enumerate(self.layers):
        #     x = layer(x)
        #     # collect feature maps
        #     if e in layer_selections:
        #         features.append(x)
        # features.append(x)
        
        # get vertex features from selected feature maps
        vert_audio_features = self.pooling(features, cur_vertices)
        return vert_audio_features

# Graph convolutional network layer definition
class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN_layer, self).__init__()
        self.weight1 = Parameter(torch.Tensor(1, in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 6. / math.sqrt((self.weight1.size(1) + self.weight1.size(0)))
        stdv *= .3
        self.weight1.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-.1, .1)

    def forward(self, features, adj, activation):
        # 0N-GCN definition, removes need for resnet layers
        features = torch.matmul(features, self.weight1)
        output = torch.matmul(adj, features[:, :, :features.shape[-1] // 3])
        output = torch.cat(
            (output, features[:, :, features.shape[-1] // 3:]), dim=-1)
        output = output + self.bias
        return activation(output)

class GCN(nn.Module):
    def __init__(self, input_features):
        super(GCN, self).__init__()
        self.num_layers = 20
        self.hidden_gcn_layers = 300

        # define output sizes for each GCN layer
        hidden_values = [input_features] + \
            [self.hidden_gcn_layers for k in range(self.num_layers - 1)] + [3]

        # define layers
        layers = []
        for i in range(self.num_layers):
            layers.append(GCN_layer(hidden_values[i], hidden_values[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, vertex_features, adj_info):
        adj = adj_info['adj']
        # iterate through GCN layers
        x = self.layers[0](vertex_features, adj, F.relu)
        for i in range(1, self.num_layers-1):
            x = self.layers[i](x, adj, F.relu)
        coords = (self.layers[-1](x, adj, lambda x: x))

        return coords

# global chart deformation class
class Encoder(nn.Module):
    def __init__(self, args, cfg, use_vision = False, use_touch = False, use_audio = False):
        super(Encoder, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_vision = use_vision
        self.use_touch = use_touch
        self.use_audio = use_audio
        self.adj_info, self.initial_positions = mdn_utils.load_mesh_vision(osp.join(osp.dirname(osp.abspath(__file__)), 'vision_sheets.obj'), \
            self.args, self.use_touch)
        
        input_size = 3  # used to determine the size of the vertex feature vector

        if self.use_vision:
            self.img_encoder = Image_Encoder().cuda()
            with torch.no_grad():
                input_size += self.img_encoder(torch.zeros(1, 1, 3).cuda(), torch.zeros(1, 3, 256, 256).cuda()).shape[-1]
        if self.use_touch:
            input_size += 1
        if self.use_audio:
            self.audio_encoder = Audio_Encoder(self.cfg, audio_num=self.args.audio_num).cuda()
            self.position_encoder = nn.Sequential(
                nn.Linear(self.args.audio_num*7,16),
                nn.LeakyReLU(),
                nn.BatchNorm1d(16),
                nn.Linear(16,8),
                nn.LeakyReLU(),
                nn.BatchNorm1d(8),
                nn.Linear(8,4),
                nn.LeakyReLU(),
                nn.BatchNorm1d(4),
                nn.Linear(4,4)
            ).cuda()
            with torch.no_grad():
                input_size += self.audio_encoder(torch.zeros(1, 1, 3).cuda(), torch.zeros(1, self.args.audio_num, 257, 257).cuda()).shape[-1]
                input_size += 4

        self.mesh_decoder = GCN(input_size).cuda()

    def normalize_vert(self, cur_vertices, gt_points):
        scale_init = torch.max(torch.sqrt(torch.sum((cur_vertices - torch.mean(cur_vertices,
                               dim=1).unsqueeze(1).cuda()) ** 2, dim=2)), dim=1).values.unsqueeze(1).unsqueeze(1).cuda()
        scale_gt = torch.max(torch.sqrt(torch.sum((gt_points.cuda() - torch.mean(gt_points,
                             dim=1).unsqueeze(1).cuda()) ** 2, dim=2)), dim=1).values.unsqueeze(1).unsqueeze(1).cuda()
        cur_vertices = cur_vertices - \
            torch.mean(cur_vertices, dim=1).unsqueeze(1).cuda()

        cur_vertices = cur_vertices / scale_init * scale_gt
        cur_vertices += torch.mean(gt_points, dim=1).unsqueeze(1).cuda()
        return cur_vertices

    def forward(self, batch, calc_loss=False):
        output = {}
        # initial data
        batch_size = batch['global_gt_points'].shape[0]
        cur_vertices = self.initial_positions.unsqueeze(
            0).expand(batch_size, -1, -1)
        cur_vertices = self.normalize_vert(cur_vertices, batch['global_gt_points'])

        size_vision_charts = cur_vertices.shape[1]

        # if using touch then append touch chart position to graph definition
        if self.use_touch:
            sheets = batch['local_gt_points'].cuda().view(batch_size, -1, 3)
            cur_vertices = torch.cat((cur_vertices, sheets), dim=1)
        # cycle thorugh deformation
        for i in range(3):
            vertex_features = cur_vertices.clone()
            # add vision features
            if self.use_vision:
                vert_img_features = self.img_encoder(cur_vertices, batch['visual_image'].cuda())
                vertex_features = torch.cat(
                    (vert_img_features, vertex_features), dim=-1)
            if self.use_audio:
                # spectrogram feature
                spectrogram = batch['spectrogram'][:,:,:,:batch['spectrogram'].shape[2]].cuda() # (bs. audio_num, 257, 257)
                vert_audio_features = self.audio_encoder(cur_vertices, spectrogram)
                # position feature
                position_info = torch.stack([torch.stack([torch.cat([j['rot'], j['pos']]) for j in item])
                                   for item in batch['camera_info']]).cuda() # (bs, impact_num, 7)
                position_info = torch.flatten(position_info, 1) # (bs, impact_num*7)
                vert_position_features = self.position_encoder(position_info).unsqueeze(1).repeat(1,vertex_features.shape[1],1) # (bs, num_points, 4)
                vertex_features = torch.cat(
                    (vert_position_features, vert_audio_features, vertex_features), dim=-1)
            # add mask for touch charts
            if self.use_touch:
                # flag corresponding to vision
                vision_chart_mask = torch.ones(
                    batch_size, size_vision_charts, 1).cuda() * 2
                touch_chart_mask = torch.ones(batch_size, self.args.touch_num).cuda().unsqueeze(-1).expand(batch_size, self.args.touch_num, 25)
                touch_chart_mask = touch_chart_mask.contiguous().view(batch_size, -1, 1)
                mask = torch.cat((vision_chart_mask, touch_chart_mask), dim=1)
                vertex_features = torch.cat((vertex_features, mask), dim=-1)
            # deform the vertex positions
            vertex_positions = self.mesh_decoder(
                vertex_features, self.adj_info)
            # avoid deforming the touch chart positions
            vertex_positions[:, size_vision_charts:] = 0
            cur_vertices = cur_vertices + vertex_positions
            
        global_pred_points = mdn_utils.batch_sample(cur_vertices, self.adj_info['faces'], self.args.global_sample_num)
        output['global_pred_points'] = global_pred_points
        if calc_loss:
            global_gt_points = batch['global_gt_points'].cuda()
            output['loss'] = chamfer_distance(global_pred_points, global_gt_points)[0]

        return output

