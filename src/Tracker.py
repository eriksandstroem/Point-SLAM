import os

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples, as_intrinsics_matrix,
                        get_tensor_from_camera, setup_seed,
                        get_selected_index_with_grad, get_rays_from_uv)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

from skimage.color import rgb2gray
from skimage import filters
from scipy.interpolate import interp1d

import wandb


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.dynamic_r_add, self.dynamic_r_query = None, None

        self.idx = slam.idx
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.renderer.sigmoid_coefficient = cfg['rendering']['sigmoid_coef_tracker']
        self.npc = slam.npc
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.exposure_feat_shared = slam.exposure_feat
        self.exposure_feat = self.exposure_feat_shared[0].clone(
        ).requires_grad_()

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.separate_LR = cfg['tracking']['separate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']
        self.sample_with_color_grad = cfg['tracking']['sample_with_color_grad']
        self.depth_limit = cfg['tracking']['depth_limit']

        self.radius_add_max = cfg['pointcloud']['radius_add_max']
        self.radius_add_min = cfg['pointcloud']['radius_add_min']
        self.radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
        self.color_grad_threshold = cfg['pointcloud']['color_grad_threshold']

        self.every_frame = cfg['mapping']['every_frame']
        self.lazy_start = cfg['mapping']['lazy_start']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.wandb = cfg['wandb']
        self.encode_exposure = cfg['model']['encode_exposure']

        self.prev_mapping_idx = -1  # init, used for mapping state
        self.frame_reader = get_dataset(
            cfg, args, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(
                                         self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device,
                                     vis_inside=cfg['tracking']['vis_inside'], total_iters=self.num_cam_iters)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def set_pipe(self, pipe):
        self.pipe = pipe

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer,
                              selected_index=None):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.
            selected_index: top color gradients pixels are pre-selected.

        Returns:
            loss (float): total loss
            color_loss (float): color loss component
            geo_loss (float): geometric loss component
        """
        device = self.device
        npc = self.npc
        H, W = self.H, self.W
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        if self.sample_with_color_grad:
            sample_size = batch_size
            cur_samples = np.random.choice(
                range(0, selected_index.shape[0]), size=sample_size, replace=False)

            index_color_grad = selected_index[cur_samples]
            i, j = np.unravel_index(index_color_grad.astype(int), (H, W))
            i, j = torch.from_numpy(j).to(device).float(
            ), torch.from_numpy(i).to(device).float()
            batch_rays_o, batch_rays_d = get_rays_from_uv(
                i, j, c2w, self.fx, self.fy, self.cx, self.cy, device)
            i, j = i.long(), j.long()
            batch_gt_depth = gt_depth[j, i]
            batch_gt_color = gt_color[j, i]

        else:
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, i, j = get_samples(
                Hedge, H-Hedge, Wedge, W - Wedge,
                batch_size,
                self.fx, self.fy, self.cx, self.cy, c2w, gt_depth, gt_color, device,
                depth_filter=True, return_index=True, depth_limit=5.0 if self.depth_limit else None)

        if self.use_dynamic_radius:
            batch_r_query = self.dynamic_r_query[j, i]
        assert torch.numel(
            batch_gt_depth) != 0, 'gt_depth after filter is empty, please check.'

        with torch.no_grad():
            inside_mask = batch_gt_depth <= torch.minimum(
                10*batch_gt_depth.median(), 1.2*torch.max(batch_gt_depth))
        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]
        batch_r_query = batch_r_query[inside_mask] if self.use_dynamic_radius else None

        ret = self.renderer.render_batch_ray(npc, self.decoders, batch_rays_d, batch_rays_o,
                                             device, stage='color',  gt_depth=batch_gt_depth,
                                             npc_geo_feats=self.npc_geo_feats,
                                             npc_col_feats=self.npc_col_feats,
                                             is_tracker=True, cloud_pos=self.cloud_pos,
                                             dynamic_r_query=batch_r_query,
                                             exposure_feat=self.exposure_feat)
        depth, uncertainty, color, _ = ret

        uncertainty = uncertainty.detach()
        nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
        # remove pixels seen as outlier
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.mean()) & (batch_gt_depth > 0)
        else:
            tmp = torch.abs(batch_gt_depth-depth)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)

        mask = mask & nan_mask

        geo_loss = torch.clamp((torch.abs(batch_gt_depth-depth) /
                                torch.sqrt(uncertainty+1e-10)), min=0.0, max=1e3)[mask].sum()
        loss = geo_loss

        color_loss = torch.abs(
            batch_gt_color - color)[mask].sum()

        if self.use_color_in_tracking:
            loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), (color_loss/mask.shape[0]).item(), (geo_loss/mask.shape[0]).item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            self.decoders = self.shared_decoders
            self.npc_geo_feats = self.npc.get_geo_feats().detach().clone()
            self.npc_col_feats = self.npc.get_col_feats().detach().clone()
            self.prev_mapping_idx = self.mapping_idx[0].clone()
            self.cloud_pos = torch.tensor(
                self.npc.cloud_pos(), device=self.device).reshape(-1, 3)
            if self.verbose:
                print('Tracker has updated the parameters from Mapper.')

    def run(self, time_string):
        setup_seed(self.cfg["setup_seed"])
        device = self.device
        self.c = {}
        scene_name = self.cfg['scene']

        if self.use_dynamic_radius:
            os.makedirs(f'{self.output}/dynamic_r_frame', exist_ok=True)
        if self.wandb and not self.gt_camera:
            wandb.init(project=self.cfg['project_name'],
                       group=f'slam_{scene_name}',
                       name='tracker_'+time_string,
                       dir=self.cfg["wandb_folder"], tags=[scene_name])

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0].item()}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if self.lazy_start:
                self.every_frame = 1 if idx <= self.lazy_start else self.cfg[
                    'mapping']['every_frame']

            if self.use_dynamic_radius:
                ratio = self.radius_query_ratio
                intensity = rgb2gray(gt_color.cpu().numpy())
                grad_y = filters.sobel_h(intensity)
                grad_x = filters.sobel_v(intensity)
                color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                color_grad_mag = np.clip(
                    color_grad_mag, 0.0, self.color_grad_threshold)
                fn_map_r_add = interp1d([0, 0.01, self.color_grad_threshold], [
                                        self.radius_add_max, self.radius_add_max, self.radius_add_min])
                fn_map_r_query = interp1d([0, 0.01, self.color_grad_threshold], [
                                          ratio*self.radius_add_max, ratio*self.radius_add_max, ratio*self.radius_add_min])
                dynamic_r_add = fn_map_r_add(color_grad_mag)
                dynamic_r_query = fn_map_r_query(color_grad_mag)
                self.dynamic_r_add, self.dynamic_r_query = torch.from_numpy(dynamic_r_add).to(
                    self.device), torch.from_numpy(dynamic_r_query).to(self.device)
                torch.save(self.dynamic_r_query,
                           f'{self.output}/dynamic_r_frame/r_query_{idx:05d}.pt')

            if self.sample_with_color_grad:
                H, W = self.H, self.W
                Wedge = self.ignore_edge_W
                Hedge = self.ignore_edge_H
                selected_index, cur_color_grad = get_selected_index_with_grad(Hedge, H-Hedge, Wedge, W-Wedge,
                                                                              self.tracking_pixels, gt_color,
                                                                              gt_depth=gt_depth, depth_limit=self.depth_limit)
            else:
                selected_index = None

            if (idx > 1 and idx % self.every_frame == 1 or (idx > 1 and self.every_frame == 1)):
                m_idx = self.pipe.recv()
                pre_c2w = self.estimate_c2w_list[idx-1].to(device)

            self.update_para_from_mapping()
            if self.encode_exposure:
                self.exposure_feat = self.exposure_feat_shared[0].clone(
                ).requires_grad_()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx <= 1 or self.gt_camera:
                c2w = gt_c2w               # remove redundant DOFs
            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                self.num_cam_iters = self.cfg['tracking']['iters']
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if torch.dot(camera_tensor[:4], gt_camera_tensor[:4]).item() < 0:
                    camera_tensor[:4] *= -1
                if self.separate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    self.quad = Variable(quad, requires_grad=True)
                    self.T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [self.T]
                    cam_para_list_quad = [self.quad]
                    optim_para_list = [{'params': cam_para_list_T, 'lr': self.cam_lr},
                                       {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}]
                    if self.encode_exposure:
                        optim_para_list.append(
                            {'params': self.exposure_feat, 'lr': 0.001})
                        optim_para_list.append(
                            {'params': self.decoders.color_decoder.mlp_exposure.parameters(), 'lr': 0.001})
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optim_para_list = [
                        {'params': cam_para_list, 'lr': self.cam_lr}]
                    if self.encode_exposure:
                        optim_para_list.append(
                            {'params': self.exposure_feat, 'lr': 0.001})
                        optim_para_list.append(
                            {'params': self.decoders.color_decoder.mlp_exposure.parameters(), 'lr': 0.001})
                optimizer_camera = torch.optim.Adam(optim_para_list)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor)
                candidate_cam_tensor = None
                current_min_loss = float(1e20)

                self.keyframe_dict = None

                for cam_iter in range(self.num_cam_iters):
                    if self.separate_LR:
                        camera_tensor = torch.cat(
                            [self.quad, self.T], 0).to(self.device)

                    loss, color_loss_pixel, geo_loss_pixel = self.optimize_cam_in_batch(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
                                                                                        optimizer_camera, selected_index=selected_index)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.detach().clone()
                    if cam_iter == self.num_cam_iters-1:
                        idx_loss_camera_tensor = torch.abs(gt_camera_tensor.to(
                            device)-candidate_cam_tensor)
                        if not self.wandb:
                            print(f'idx:{idx}, re-rendering loss: {initial_loss:.2f}->{current_min_loss:.2f}, ' +
                                  f'camera_quad_error: {initial_loss_camera_tensor[:4].mean().item():.4f}->{idx_loss_camera_tensor[:4].mean().item():.4f}, '
                                  + f'camera_pos_error: {initial_loss_camera_tensor[-3:].mean().item():.4f}->{idx_loss_camera_tensor[-3:].mean().item():.4f}')
                        if self.wandb and not self.gt_camera:
                            wandb.log({'camera_quad_error': idx_loss_camera_tensor[:4].mean().item(),
                                       'camera_pos_error': idx_loss_camera_tensor[-3:].mean().item(),
                                       'color_loss_tracker': color_loss_pixel,
                                       'geo_loss_tracker': geo_loss_pixel,
                                       'idx_track': int(idx.item())})
                    else:
                        if cam_iter % 20 == 0:
                            if not self.wandb:
                                print(
                                    f'iter: {cam_iter}, camera tensor error: {loss_camera_tensor:.4f}')

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor,
                        self.npc, self.decoders, self.npc_geo_feats, self.npc_col_feats,
                        dynamic_r_query=self.dynamic_r_query, cloud_pos=self.cloud_pos,
                        exposure_feat=self.exposure_feat)

                bottom = torch.tensor(
                    [0, 0, 0, 1.0], device=self.device).reshape(1, 4)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.detach().clone())
                c2w = torch.cat([c2w, bottom], dim=0)

            if idx > 0 and idx % self.every_frame == 0 or idx == self.n_img - 1:
                self.pipe.send(idx)

            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            if self.encode_exposure:
                self.exposure_feat_shared[0] = self.exposure_feat.detach(
                ).clone()
            self.idx[0] = idx

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

        if self.wandb and not self.gt_camera:
            wandb.finish()
