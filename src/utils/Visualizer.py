import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
import wandb
import cv2


class Visualizer(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0', wandb=False,
                 vis_inside=False, total_iters=None, img_dir=None):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.crop_edge = self.renderer.crop_edge
        self.inside_freq = inside_freq
        self.wandb = wandb
        self.vis_inside = vis_inside
        self.total_iters = total_iters
        self.img_dir = img_dir
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def vis_value_only(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, npc,
                       decoders, npc_geo_feats, npc_col_feats, freq_override=False,
                       dynamic_r_query=None, cloud_pos=None, exposure_feat=None):
        """
        return rendered depth and color map only
        """
        with torch.no_grad():
            if freq_override or (idx % self.freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                        torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.detach().clone())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    npc,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth, npc_geo_feats=npc_geo_feats,
                    npc_col_feats=npc_col_feats,
                    dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos,
                    exposure_feat=exposure_feat)
                return depth, color

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, npc,
            decoders, npc_geo_feats, npc_col_feats, freq_override=False,
            dynamic_r_query=None, cloud_pos=None, exposure_feat=None,
            cur_total_iters=None, save_rendered_image=False):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            npc (): neural point cloud.
            decoders (nn.module): decoders.
            npc_geo_feats (tensor): point cloud geometry features, cloned from npc. Optimizable during mapping.
            npc_col_feats (tensor): point cloud color features. Optimizable during mapping.
            freq_override (bool): call vis() at will
            dynamic_r_query (tensor, optional): if use dynamic query, for every ray, its query radius is different.
            cloud_pos (tensor): positions of all point cloud features, used only when tracker calls.
            exposure_feat (tensor, optional): whether to render with an exposure feature vector. All rays have the same
            exposure feature vector.
            cur_total_iters (int): number of iterations done when saving
            save_rendered_image (bool): whether to save the rgb image in separate folder apart from the standard visualization
        """
        with torch.no_grad():
            if self.vis_inside:
                conditions = (idx > 0 and (idx % self.freq == 0) and (
                    (iter % self.inside_freq == 0) or (iter == self.total_iters-1))) or freq_override
            else:
                conditions = (idx > 0 and (idx % self.freq == 0) and (
                    iter == self.total_iters-1)) or freq_override
            if conditions:
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                        torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.detach().clone())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    npc,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth, npc_geo_feats=npc_geo_feats,
                    npc_col_feats=npc_col_feats,
                    dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos,
                    exposure_feat=exposure_feat)
                if save_rendered_image and self.img_dir is not None:
                    img = cv2.cvtColor(color.cpu().numpy()
                                       * 255, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(
                        f'{self.img_dir}', f'frame_{idx:05d}.png'), img)

                depth_np = depth.detach().cpu().numpy()
                color = torch.round(color*255.0)/255.0
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                depth_residual = np.clip(depth_residual, 0.0, 0.05)

                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Rendered Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma")
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Rendered RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                fig_name = f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg'
                plt.savefig(fig_name, dpi=300,
                            bbox_inches='tight', pad_inches=0.1)
                if 'mapping' in self.vis_dir and self.wandb:
                    wandb.log(
                        ({f'Mapping_{idx:05d}_{iter:04d}': wandb.Image(fig_name)}))
                plt.clf()
                plt.close()

                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter if cur_total_iters is None else cur_total_iters:04d}.jpg')
