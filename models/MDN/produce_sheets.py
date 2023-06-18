import os
import os.path as osp
import sys
import mdn_utils
import torch
import math
import numpy as np
import torch.optim as optim
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R

local_gt_points_location = '../../../DATA_new/local_gt_points/'
local_gt_sheets_location = '../../../DATA_new/local_gt_sheets/'
camera_info_location = '../../../DATA_new/camera_info/'

# load the #touch local point cloud of obj
def load_local_gt_points(obj, touch):
    local_gt_points = np.load(
        osp.join(local_gt_points_location, obj, f'{touch}.npy'))
    local_gt_points = torch.FloatTensor(local_gt_points)
    return local_gt_points

# load the #touch camera information (position + rotation) of obj
def load_camera_info(obj, touch):
    data = np.load(osp.join(camera_info_location, obj, f'{touch}.npy'),allow_pickle=True).item()
    camera_info = {}
    rot = data['cam_rot']
    pos = data['cam_pos']
    rot_M = R.from_euler('xyz', rot, degrees=False).as_matrix()
    rot_q = R.from_matrix(rot_M).as_quat()
    camera_info['rot'] = torch.FloatTensor(rot_q)
    camera_info['rot_M'] = torch.FloatTensor(rot_M)
    camera_info['pos'] = torch.FloatTensor(pos)
    return camera_info

with open('./sheet_remain.txt') as f:
    remain=f.readlines()
start_idx = int(remain[0])
remain.remove(remain[0])
with open('./sheet_remain.txt','w') as f:
    f.writelines(remain)
print("Start_idx: ",start_idx)

verts, faces = mdn_utils.load_mesh_touch('initial_sheet.obj')
faces = faces.cuda()
for obj in trange(start_idx, start_idx+1):
    cur_local_gt_sheets_location = osp.join(local_gt_sheets_location,str(obj))
    os.makedirs(cur_local_gt_sheets_location,exist_ok=True)
    for touch in trange(1, 101, leave=False):
        torch.cuda.empty_cache()
        print('mem: ',torch.cuda.memory_allocated())
        if osp.exists(osp.join(local_gt_points_location, str(obj), '{}.npy'.format(touch))):
            local_gt_points = load_local_gt_points(str(obj), str(touch)).cuda()
            camera_info = load_camera_info(str(obj), str(touch))
            # make initial mesh match touch sensor when touch occurred
            initial = verts.clone().unsqueeze(0).cuda()
            pos = camera_info['pos'].cuda().view(1, -1)
            rot = camera_info['rot_M'].cuda().view(1, 3, 3)
            initial = torch.bmm(rot, initial.permute(0, 2, 1)).permute(0, 2, 1)
            initial += pos.view(1, 1, 3)
            initial = initial[0]
            updates = torch.zeros(verts.shape, requires_grad=True, device="cuda")
            optimizer = optim.Adam([updates], lr=1)
            last_improvement = 0
            best_loss = 1e10
            i=0
            while True:
                # update
                optimizer.zero_grad()
                verts = initial + updates
                # losses
                surf_loss = mdn_utils.chamfer_distance(verts.unsqueeze(0), faces,\
                            local_gt_points.unsqueeze(0), num = 4000)
                edge_lengths = mdn_utils.batch_calc_edge(verts.unsqueeze(0), faces)
                loss =  9000 * surf_loss + 70 * edge_lengths
                # # optimize
                loss.backward(retain_graph=True)
                optimizer.step()
                # check results
                if loss < 1e-3:
                    break
                if (best_loss - loss) / best_loss > 1e-3:
                    best_loss = loss
                    best_verts = verts.clone()
                    last_improvement = 0
                else:
                    last_improvement += 1
                    if last_improvement > 100:
                        break
            np.save(osp.join(cur_local_gt_sheets_location,"{}.npy".format(touch)), best_verts.data.cpu().numpy())