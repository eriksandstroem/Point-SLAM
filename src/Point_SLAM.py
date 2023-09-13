import os

import torch
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Renderer import Renderer
from src.neural_point import NeuralPointCloud
from src.common import setup_seed

torch.multiprocessing.set_sharing_strategy('file_system')


class Point_SLAM():
    """
    POINT_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args, share_npc=True, share_decoders=True, time_string=None):

        self.cfg = cfg
        self.args = args

        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']

        self.time_string = time_string

        if args.output is None:
            cfg["data"]["output"] = os.path.join(
                cfg["data"]["output"], time_string) if time_string else cfg["data"]["output"]
            self.output = cfg['data']['output']
        else:
            args.output = os.path.join(
                args.output, time_string) if time_string else args.output
            cfg['data']['output'] = args.output
            self.output = args.output

        if args.wandb:
            cfg['wandb'] = True
        elif args.no_wandb:
            cfg['wandb'] = False
        if args.input_folder:
            cfg["data"]["input_folder"] = args.input_folder
    
        self.cfg = cfg

        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        if cfg['mapping']['save_rendered_image']:
            os.makedirs(f'{self.output}/rendered_image', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg)
        self.shared_decoders = model

        self.load_pretrain(cfg)

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.exposure_feat = torch.zeros((1, cfg['model']['exposure_dim'])).normal_(
            mean=0, std=0.01).to(self.cfg['mapping']['device'])
        self.exposure_feat.share_memory_()
        if share_decoders:
            self.shared_decoders = self.shared_decoders.to(
                self.cfg['mapping']['device'])
            self.shared_decoders.share_memory()

        if share_npc:
            BaseManager.register('NeuralPointCloud', NeuralPointCloud)
            manager = BaseManager()
            manager.start()
            self.npc = manager.NeuralPointCloud(cfg)
        else:
            self.npc = NeuralPointCloud(cfg)

        self.renderer = Renderer(cfg, args, self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

    def print_output_desc(self):
        print("")
        print(f"⭐️ INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"⭐️ INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"⭐️ INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"⭐️ INFO: The mesh can be found under {self.output}/mesh/")
        print(
            f"⭐️ INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'],
                          map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}
        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.geo_decoder.load_state_dict(
            middle_dict, strict=False)

    def tracking(self, rank, time_string, pipe):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        self.tracker.set_pipe(pipe)
        first = pipe.recv()
        self.tracker.run(time_string)

    def mapping(self, rank, time_string, pipe):
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.set_pipe(pipe)
        self.mapper.run(time_string)

    def run(self):
        """
        Dispatch Threads. # this func, when called, act as main process
        """
        setup_seed(self.cfg["setup_seed"])

        processes = []
        m_pipe, t_pipe = mp.Pipe()
        for rank in range(2):
            if rank == 0:
                p = mp.Process(name='tracker', target=self.tracking,
                               args=(rank, self.time_string, t_pipe))
            elif rank == 1:
                p = mp.Process(name='mapper', target=self.mapping,
                               args=(rank, self.time_string, m_pipe))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == '__main__':
    pass
