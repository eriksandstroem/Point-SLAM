import os

import torch


class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, cfg, args, mapper
                 ):
        self.verbose = mapper.verbose
        self.ckptsdir = mapper.ckptsdir
        self.gt_c2w_list = mapper.gt_c2w_list
        self.estimate_c2w_list = mapper.estimate_c2w_list
        self.decoders = mapper.decoders

    def log(self, idx, keyframe_dict, keyframe_list, selected_keyframes, npc, exposure_feat=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'geo_feats': npc.get_geo_feats(),
            'col_feats': npc.get_col_feats(),
            'cloud_pos': npc.cloud_pos(),
            'pts_num': npc.pts_num(),
            'input_pos': npc.input_pos(),
            'input_rgb': npc.input_rgb(),

            'decoder_state_dict': self.decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'keyframe_dict': keyframe_dict,
            'selected_keyframes': selected_keyframes,
            'idx': idx,
            "exposure_feat_all": torch.stack(exposure_feat, dim=0)
            if exposure_feat is not None
            else None,
        }, path)

        if self.verbose:
            print('Saved checkpoints at', path)
