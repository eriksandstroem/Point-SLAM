import sys
from scipy.interpolate import interp1d
from skimage import filters
from skimage.color import rgb2gray
import subprocess
import os
import random
import argparse
import numpy as np
import torch
import open3d as o3d
import warnings
import trimesh
import cv2

from torch.utils.data import Dataset, DataLoader
sys.path.append('.')
from src import config
from src.Point_SLAM import Point_SLAM
from src.utils.Visualizer import Visualizer
from src.utils.datasets import get_dataset



class DepthImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.depth_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('depth_')])
        self.image_files = sorted([os.path.join(root_dir, f) for f in os.listdir(
            root_dir) if f.startswith('color_')])

        indices = []
        for depth_file in self.depth_files:
            base, ext = os.path.splitext(depth_file)
            index = int(base[-5:])
            indices.append(index)
        self.indices = indices

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, idx):
        depth = np.load(self.depth_files[idx])
        image = np.load(self.image_files[idx])

        if self.transform:
            depth = self.transform(depth)
            image = self.transform(image)

        return depth, image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_neural_point_cloud(slam, ckpt, device, use_exposure=False):

    slam.npc._cloud_pos = ckpt['cloud_pos']
    slam.npc._input_pos = ckpt['input_pos']
    slam.npc._input_rgb = ckpt['input_rgb']
    slam.npc._pts_num = len(ckpt['cloud_pos'])
    slam.npc.geo_feats = ckpt['geo_feats'].to(device)
    slam.npc.col_feats = ckpt['col_feats'].to(device)
    if use_exposure:
        assert 'exposure_feat_all' in ckpt.keys(
        ), 'Please check if exposure feature is encoded.'
        slam.mapper.exposure_feat_all = ckpt['exposure_feat_all'].to(device)

    cloud_pos = torch.tensor(ckpt['cloud_pos'], device=device)
    slam.npc.index_train(cloud_pos)
    slam.npc.index.add(cloud_pos)

    print(
        f'Successfully loaded neural point cloud, {slam.npc.index.ntotal} points in total.')


def load_ckpt(cfg, slam):
    """
    Saves mesh of already reconstructed model from checkpoint file. Makes it 
    possible to remesh reconstructions with different settings.
    """

    assert cfg['mapping']['save_selected_keyframes_info'], 'Please save keyframes info to help run this code.'

    ckptsdir = f'{slam.output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('\nGet ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location='cpu')
        else:
            raise ValueError(f'Check point directory {ckptsdir} is empty.')
    else:
        raise ValueError(f'Check point directory {ckptsdir} not found.')

    return ckpt


def clean_mesh(mesh):
    mesh_tri = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(
        mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    components = trimesh.graph.connected_components(
        edges=mesh_tri.edges_sorted)

    min_len = 100
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices = []
    new_faces = []
    new_colors = []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]

        # Create a mapping from old vertex indices to new vertex indices
        index_mapping = {old_idx: vertex_count +
                         new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)

        # Select faces that are part of the current connected component and update vertex indices
        faces_in_component = mesh_tri.faces[np.any(
            np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.remove_degenerate_faces()
    cleaned_mesh_tri.remove_duplicate_faces()
    print(
        f'Mesh cleaning (before/after), vertices: {len(mesh_tri.vertices)}/{len(cleaned_mesh_tri.vertices)}, faces: {len(mesh_tri.faces)}/{len(cleaned_mesh_tri.faces)}')

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces)
    )
    vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[
        :, :3] / 255.0
    cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors.astype(np.float64))

    return cleaned_mesh


def main():
    parser = argparse.ArgumentParser(
        description="Configs for Point-SLAM."
    )
    parser.add_argument(
        "config", type=str, help="Path to config file.",
    )
    parser.add_argument("--input_folder", type=str,
                        help="input folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--output", type=str,
                        help="output folder, this have higher priority, can overwrite the one in config file.",
                        )
    parser.add_argument("--name", type=str,
                        help="specify the name of the mesh",
                        )
    parser.add_argument("--no_render", default=False, action='store_true',
                        help="if to render frames from checkpoint for constructing the mesh.",
                        )
    parser.add_argument("--exposure_avail", default=False, action='store_true',
                        help="if the exposure information is available for rendering.",
                        )
    parser.add_argument("-s", "--silent", default=False, action='store_true',
                        help="if to print status message.",
                        )
    parser.add_argument("--no_eval", default=False, action='store_true',
                        help="if to evaluate the mesh by 2d and 3d metrics.",
                        )

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--clean', default=False,
                        action='store_true', help='Enable mesh cleaning')

    args = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU required for reconstruction.'
    cfg = config.load_config(args.config, "configs/point_slam.yaml")
    device = cfg['mapping']['device']

    # define variables for dynamic query radius computation
    radius_add_max = cfg['pointcloud']['radius_add_max']
    radius_add_min = cfg['pointcloud']['radius_add_min']
    radius_query_ratio = cfg['pointcloud']['radius_query_ratio']
    color_grad_threshold = cfg['pointcloud']['color_grad_threshold']

    slam = Point_SLAM(cfg, args, share_npc=False,
                      share_decoders=False if args.no_render else True)
    slam.output = cfg['data']['output'] if args.output is None else args.output
    ckpt = load_ckpt(cfg, slam)

    render_frame = not args.no_render
    use_exposure = args.exposure_avail
    frame_reader = get_dataset(cfg, args, device=device)
    if render_frame:
        load_neural_point_cloud(slam, ckpt, device, use_exposure=use_exposure)
        idx = 0
        frame_cnt = 0

        try:
            slam.shared_decoders.load_state_dict(ckpt['decoder_state_dict'])
            if not args.silent:
                print('Successfully loaded decoders.')
        except Exception as e:
            raise ValueError(f'Cannot load decoders: {e}')

        visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                vis_dir=os.path.join(slam.output, 'rendered_every_frame'), renderer=slam.renderer,
                                verbose=slam.verbose, device=device, wandb=False)

        cloud_pos_tensor = torch.tensor(
            slam.npc.cloud_pos(), device=device)

        if not args.silent:
            print('Starting to render frames...')
        last_idx = (ckpt['idx']+1) if (ckpt['idx'] +
                                       1) < len(frame_reader) else len(frame_reader)
        while idx < last_idx:
            _, gt_color, gt_depth, gt_c2w = frame_reader[idx]
            cur_c2w = ckpt['estimate_c2w_list'][idx].to(device)

            if use_exposure:
                try:
                    state_dict = torch.load(f'{slam.output}/ckpts/color_decoder/{idx:05}.pt',
                                            map_location=device)
                    slam.shared_decoders.color_decoder.load_state_dict(
                        state_dict)
                except Exception as e:
                    print(e)
                    raise ValueError(
                        f'Cannot load per mapping-frame color decoder at frame {idx}.')

            ratio = radius_query_ratio
            intensity = rgb2gray(gt_color.cpu().numpy())
            grad_y = filters.sobel_h(intensity)
            grad_x = filters.sobel_v(intensity)
            color_grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # range 0~1
            color_grad_mag = np.clip(
                color_grad_mag, 0.0, color_grad_threshold)  # range 0~1

            fn_map_r_query = interp1d([0, 0.01, color_grad_threshold], [
                ratio*radius_add_max, ratio*radius_add_max, ratio*radius_add_min])
            dynamic_r_query = fn_map_r_query(color_grad_mag)
            dynamic_r_query = torch.from_numpy(dynamic_r_query).to(device)

            cur_frame_depth, cur_frame_color = visualizer.vis_value_only(idx, 0, gt_depth, gt_color, cur_c2w, slam.npc, slam.shared_decoders,
                                                                         slam.npc.geo_feats, slam.npc.col_feats, freq_override=True,
                                                                         dynamic_r_query=dynamic_r_query, cloud_pos=cloud_pos_tensor,
                                                                         exposure_feat=slam.mapper.exposure_feat_all[
                                                                             idx // cfg["mapping"]["every_frame"]
                                                                         ].to(device) if use_exposure else None)
            np.save(f'{slam.output}/rendered_every_frame/depth_{idx:05d}',
                    cur_frame_depth.cpu().numpy())
            np.save(f'{slam.output}/rendered_every_frame/color_{idx:05d}',
                    cur_frame_color.cpu().numpy())

            idx += cfg['mapping']['every_frame']
            frame_cnt += 1
            if idx % 400 == 0:
                print(f'{idx}...')
        if not args.silent:
            print(f'Finished rendering {frame_cnt} frames.')

    dataset = DepthImageDataset(root_dir=slam.output+'/rendered_every_frame')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    scene_name = cfg["scene"]
    mesh_name = f'{scene_name}_pred_mesh.ply' if args.name is None else args.name
    mesh_out_file = f'{slam.output}/mesh/{mesh_name}'

    H, W, fx, fy, cx, cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    if not args.silent:
        print('Starting to integrate the mesh...')
    # address the misalignment in open3d marching cubes
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                         scale / 512.0, -2.5 * scale / 512.0)

    os.makedirs(f'{slam.output}/mesh/mid_mesh', exist_ok=True)

    for i, (depth, color) in enumerate(dataloader):
        index = dataset.indices[i]
        # load the gt depth from the sensor to filter the rendered depth map
        _, gt_color, gt_depth, gt_c2w = frame_reader[cfg['mapping']
                                                     ['every_frame']*i]
        gt_depth = gt_depth.cpu().numpy()
        depth = depth[0].cpu().numpy()
        # the rendered depth map is not accurate in areas where no sensor
        # depth was observed. Set the rendered pixels to 0 where the
        # no sensor depth exists.
        depth[gt_depth == 0] = 0
        color = color[0].cpu().numpy()
        c2w = ckpt['estimate_c2w_list'][index].cpu().numpy()

        c2w[:3, 1] *= -1.0
        c2w[:3, 2] *= -1.0
        w2c = np.linalg.inv(c2w)

        depth = o3d.geometry.Image(depth.astype(np.float32))
        color = o3d.geometry.Image(
            np.array((np.clip(color, 0.0, 1.0)*255.0).astype(np.uint8)))

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=30,
            convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, w2c)

        if i > 0 and cfg["meshing"]["mesh_freq"] > 0 and (i % cfg["meshing"]["mesh_freq"]) == 0:
            o3d_mesh = volume.extract_triangle_mesh()
            o3d_mesh = o3d_mesh.translate(compensate_vector)
            if args.clean or cfg['dataset'] != 'replica':
                o3d_mesh = clean_mesh(o3d_mesh)
            o3d.io.write_triangle_mesh(
                f"{slam.output}/mesh/mid_mesh/frame_{cfg['mapping']['every_frame']*i}_mesh.ply", o3d_mesh)
            print(
                f"saved intermediate mesh until frame {cfg['mapping']['every_frame']*i}.")

    o3d_mesh = volume.extract_triangle_mesh()
    np.save(os.path.join(f'{slam.output}/mesh',
            'vertices_pos.npy'), np.asarray(o3d_mesh.vertices))
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    if args.clean or cfg['dataset'] != 'replica':
        o3d_mesh = clean_mesh(o3d_mesh)

    o3d.io.write_triangle_mesh(mesh_out_file, o3d_mesh)
    if not args.silent:
        print('🕹️ Meshing finished.')

    eval_recon = not args.no_eval
    if eval_recon:
        try:
            if cfg['dataset'] == 'replica':
                print('Evaluating...')
                result_recon_obj = subprocess.run(['python', '-u', 'src/tools/eval_recon.py', '--rec_mesh',
                                                   mesh_out_file,
                                                   '--gt_mesh', f'cull_replica_mesh/{scene_name}.ply', '-3d', '-2d'],
                                                  text=True, check=True, capture_output=True)
                result_recon = result_recon_obj.stdout
                print(result_recon)
                print('✨ Successfully evaluated 3D reconstruction.')
            else:
                print(
                    'Current dataset is not supported for 3D reconstruction evaluation.')
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            print('Failed to evaluate 3D reconstruction.')


if __name__ == "__main__":
    main()
