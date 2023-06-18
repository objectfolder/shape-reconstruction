import os
import os.path as osp
import sys
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim

import utils.metrics as metrics
import utils.meters as meters
from models.build import build as build_model
from dataset.build import build as build_dataset


class Engine():
    def __init__(self, args, cfg):
        args.touch_num = args.impact_num
        args.audio_num = args.impact_num
        self.args = args
        self.cfg = cfg
        # set seeds
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        # build dataloaders
        self.train_loader, self.val_loader, self.test_loader = build_dataset(self.args)
        # build model & optimizer
        self.model, self.optimizer = build_model(self.args, self.cfg)
        self.model.cuda()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.2)
        # experiment dir
        self.save_viz = True
        self.exp_dir = osp.join('./exp',self.args.exp)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.exp_viz_dir = osp.join(self.exp_dir, 'viz')
        os.makedirs(self.exp_viz_dir, exist_ok=True)
        
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = meters.AverageMeter()
        for i, batch in tqdm(enumerate(self.train_loader), leave = False):
            self.optimizer.zero_grad()
            output = self.model(batch, calc_loss = True)
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            
            epoch_loss.update(10000. * loss.item(), self.args.batch_size)
            if i % 10 == 0:
                message = f'Epoch: {epoch}, loss: {epoch_loss.avg:.2f}'
                tqdm.write(message)
        tqdm.write("Finish Training Epoch {}, loss = {:.2f}".format(epoch, epoch_loss.avg))
                
    @torch.no_grad()
    def eval_epoch(self, test = False):
        self.model.eval()
        epoch_chamfer_dist = meters.AverageMeter()
        data_loader = self.test_loader if test else self.val_loader
        for iter, batch in tqdm(enumerate(data_loader), leave = False):
            output = self.model(batch)
            global_gt_points = 100 * batch['global_gt_points'].detach().cpu().numpy()*batch['scale'].detach().cpu().numpy()
            global_pred_points = 100 * output['global_pred_points'].detach().cpu().numpy()*batch['scale'].detach().cpu().numpy()
            epoch_chamfer_dist.update(metrics.TrimeshChamferDistance(
                global_gt_points, global_pred_points), self.args.batch_size)
            if iter%100 == 0:
                np.save(osp.join(self.exp_viz_dir,f'gt_iter{iter}.npy'),global_gt_points[:,:1024])
                np.save(osp.join(self.exp_viz_dir,f'pred_iter{iter}.npy'),global_pred_points[:,:1024])
        return epoch_chamfer_dist.avg
            
    def train(self):
        bst_epoch_chamfer_dist = 1e8
        for epoch in range(self.args.epochs):
            print("Start Validation Epoch {}".format(epoch))
            epoch_chamfer_dist = self.eval_epoch()
            print("Finish Validation Epoch {}, Chamfer Distance = {:.4f} (cm)".format(epoch, epoch_chamfer_dist))
            if epoch_chamfer_dist < bst_epoch_chamfer_dist:
                print("New best Chamfer Distance {:.2f} reached, saving best model".format(epoch_chamfer_dist))
                bst_epoch_chamfer_dist = epoch_chamfer_dist
                torch.save(self.model.state_dict(), osp.join(self.exp_dir,'bst.pth'))
            torch.save(self.model.state_dict(), osp.join(self.exp_dir,'latest.pth'))
            print("Start Training Epoch {}".format(epoch))
            self.train_epoch(epoch)
            self.scheduler.step()
        print("Finish Training Process")
        
    def test(self):
        print("Start Testing")
        print("Loading best model from {}".format(osp.join(self.exp_dir,'bst.pth')))
        self.model.load_state_dict(torch.load(osp.join(self.exp_dir, 'bst.pth')))
        test_chamfer_dist = self.eval_epoch(test = True)
        json.dump({"Test Result":test_chamfer_dist}, open(osp.join(self.exp_dir, 'result.json'),'w'))
        print("Finish Testing, Chamfer Distance = {:.4f} (cm)".format(test_chamfer_dist))
        
    def __call__(self):
        if not self.args.eval:
            self.train()
        self.test()