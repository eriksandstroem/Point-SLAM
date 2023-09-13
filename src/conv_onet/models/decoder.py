import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=False, concat=True):
        super().__init__()
        self.concat = concat
        self.mapping_size = mapping_size
        self.scale = scale
        self.learnable = learnable
        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = (2*math.pi*x) @ self._B.to(x.device)
        if self.concat:
            return torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
        else:
            return torch.sin(x)


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self, mapping_size=3):
        super().__init__()
        self.mapping_size = mapping_size

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP_geometry(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
        use_view_direction (bool): whether to use view direction or not.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False, sample_mode='bilinear',
                 skips=[2], pos_embedding_method='fourier',
                 concat_feature=False, use_view_direction=False):
        super().__init__()
        self.feat_name = 'geometry_feat'
        self.c_dim = c_dim
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.N_surface = cfg['rendering']['N_surface']
        self.use_view_direction = use_view_direction

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            # the input dimension is always 3
            self.embedder = GaussianFourierFeatureTransform(
                3, mapping_size=embedding_size, scale=25, concat=False, learnable=True)
            if self.use_view_direction:
                self.embedder_view_direction = GaussianFourierFeatureTransform(
                    3, mapping_size=embedding_size, scale=25)
            self.embedder_rel_pos = GaussianFourierFeatureTransform(
                3, mapping_size=10, scale=32, learnable=True)
        self.mlp_col_neighbor = MLP_col_neighbor(
            self.c_dim, 2*self.embedder_rel_pos.mapping_size, hidden_size)

        # xyz coord. -> embedding size
        embedding_input = embedding_size
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 1, activation="relu")

        if not leaky:
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None,
                           dynamic_r_query=None):
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.detach().clone(),
                                                      step='query',
                                                      dynamic_radius=dynamic_r_query)

        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        radius_query_bound = npc.get_radius_query(
        )**2 if not self.use_dynamic_radius else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D to propagate gradients to the camera extrinsics
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            D = D.reshape(-1, nn_num)

        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            weights = 1.0/(D+1e-10)
        else:
            # try to avoid over-smoothing by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        weights[D > radius_query_bound] = 0.

        # (n_points, nn_num=8, 1)
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearest nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]  # (n_points, nn_num=8, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        # points with no neighbors are given a random feature vector
        # rays that have no neighbors are thus rendered with random feature vectors for depth
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, has_neighbors  # (N_point,c_dim), mask for pts

    def forward(self, p, npc, npc_geo_feats, pts_num=16, is_tracker=False, cloud_pos=None,
                pts_views_d=None, dynamic_r_query=None):
        """forward method of geometric decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_geo_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether called by tracker. Defaults to False.
            cloud_pos (tensor, optional): point cloud position. 
            pts_views_d (tensor): viweing directions
            dynamic_r_query (tensor, optional): if enabled dynamic radius, query radius for every pixel will be different.

        Returns:
            out (tensor): occupancies for the points p
            valid_ray_mask (bool): boolen tensor. True if at least half of all points along the ray have neighbors
            has_neighbors (bool): boolean tensor. False if at least two neighbors were not found for the point in question
        """

        c, has_neighbors = self.get_feature_at_pos(
            npc, p, npc_geo_feats, is_tracker, cloud_pos, dynamic_r_query=dynamic_r_query)  # get (N,c_dim), e.g. (N,32)

        # ray is not close to the current npc, choose bar here
        # a ray is considered valid if at least half of all points along the ray have neighbors.
        valid_ray_mask = ~(
            torch.sum(has_neighbors.view(-1, pts_num), 1) < int(self.N_surface/2+1))

        p = p.float().reshape(1, -1, 3)

        embedded_pts = self.embedder(p)
        embedded_input = embedded_pts

        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)

        # (N,1)->(N,) for occupancy
        out = out.squeeze(-1)
        return out, valid_ray_mask, has_neighbors


class MLP_col_neighbor(nn.Module):
    # F_theta network in paper
    def __init__(self, c_dim, embedding_size_rel, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(c_dim + embedding_size_rel, hidden_size)
        self.linear2 = nn.Linear(hidden_size, c_dim)
        self.act_fn = nn.Softplus(beta=100)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class MLP_exposure(nn.Module):
    # Exposure compensation MLP
    def __init__(self, latent_dim, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 12)
        self.act_fn = nn.Softplus(beta=100)

        init.normal_(self.linear1.weight, mean=0, std=0.01)
        init.normal_(self.linear2.weight, mean=0, std=0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class MLP_color(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
        use_view_direction (bool): whether to use view direction.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128, n_blocks=5, leaky=False, sample_mode='bilinear',
                 skips=[2], pos_embedding_method='fourier',
                 concat_feature=False, use_view_direction=False):
        super().__init__()
        self.feat_name = 'color_feat'
        self.c_dim = c_dim
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.weighting = cfg['pointcloud']['nn_weighting']
        self.min_nn_num = cfg['pointcloud']['min_nn_num']
        self.use_dynamic_radius = cfg['use_dynamic_radius']
        self.N_surface = cfg['rendering']['N_surface']
        self.use_view_direction = use_view_direction
        self.encode_rel_pos_in_col = cfg['model']['encode_rel_pos_in_col']
        self.encode_exposure = cfg['model']['encode_exposure']
        self.encode_viewd = cfg['model']['encode_viewd']

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 20
            # the input dimension is always 3
            self.embedder = GaussianFourierFeatureTransform(
                3, mapping_size=embedding_size, scale=32)
            if self.use_view_direction:
                if self.encode_viewd:
                    self.embedder_view_direction = GaussianFourierFeatureTransform(
                        3, mapping_size=embedding_size, scale=32)
                else:
                    self.embedder_view_direction = Same(mapping_size=3)
            self.embedder_rel_pos = GaussianFourierFeatureTransform(
                3, mapping_size=10, scale=32, learnable=True)
        self.mlp_col_neighbor = MLP_col_neighbor(
            self.c_dim, 2*self.embedder_rel_pos.mapping_size, hidden_size)
        if self.encode_exposure:
            self.mlp_exposure = MLP_exposure(
                cfg['model']['exposure_dim'], hidden_size)

        # xyz coord. -> embedding size
        embedding_input = 2*embedding_size
        if self.use_view_direction:
            embedding_input += (2 if self.encode_viewd else 1) * \
                self.embedder_view_direction.mapping_size
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_input, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_input, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
            hidden_size, 3, activation="linear")

        if not leaky:
            self.actvn = torch.nn.Softplus(beta=100)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def get_feature_at_pos(self, npc, p, npc_feats, is_tracker=False, cloud_pos=None,
                           dynamic_r_query=None):
        assert torch.is_tensor(
            p), 'point locations for get_feature_at_pos should be tensor.'
        device = p.device
        p = p.reshape(-1, 3)
        D, I, neighbor_num = npc.find_neighbors_faiss(p.detach().clone(),
                                                      step='query',
                                                      dynamic_radius=dynamic_r_query)
        # faiss returns "D" in the form of squared distances. Thus we compare D to the squared radius
        radius_query_bound = npc.get_radius_query(
        )**2 if not self.use_dynamic_radius else dynamic_r_query.reshape(-1, 1)**2
        if is_tracker:
            # re-calculate D to propagate gradients to the camera extrinsics
            nn_num = D.shape[1]
            D = torch.sum(torch.square(
                cloud_pos[I]-p.reshape(-1, 1, 3)), dim=-1)
            D = D.reshape(-1, nn_num)

        has_neighbors = neighbor_num > self.min_nn_num-1

        if self.weighting == 'distance':
            weights = 1.0/(D+1e-10)
        else:
            # try to avoid over-smoothing by e^(-x)
            weights = torch.exp(-20*torch.sqrt(D))
        weights[D > radius_query_bound] = 0.
        weights = F.normalize(weights, p=1, dim=1).unsqueeze(-1)

        # use fixed num of nearest nn
        # select neighbors within range, then interpolate feature by inverse distance weighting
        neighbor_feats = npc_feats[I]             # (n_points, nn_num=8, c_dim)
        if self.encode_rel_pos_in_col:
            neighbor_pos = cloud_pos[I]  # (N,nn_num,3)
            neighbor_rel_pos = neighbor_pos - p[:, None, :]
            embedding_rel_pos = self.embedder_rel_pos(
                neighbor_rel_pos.reshape(-1, 3))             # (N, nn_num, 40)
            neighbor_feats = torch.cat([embedding_rel_pos.reshape(neighbor_pos.shape[0], -1, self.embedder_rel_pos.mapping_size*2),
                                        neighbor_feats], dim=-1)  # (N, nn_num, 40+c_dim)
            neighbor_feats = self.mlp_col_neighbor(
                neighbor_feats)                  # (N, nn_num, c_dim)

        c = weights * neighbor_feats
        c = c.sum(axis=1).reshape(-1, self.c_dim)
        # points with no neighbors are given a random feature vector
        # rays that have no neighbors are thus rendered with random feature vectors for color
        c[~has_neighbors] = torch.zeros(
            [self.c_dim], device=device).normal_(mean=0, std=0.01)

        return c, has_neighbors  # (N_point,c_dim), mask for pts

    def forward(self, p, npc, npc_col_feats, is_tracker=False, cloud_pos=None, pts_views_d=None, dynamic_r_query=None, exposure_feat=None):
        """forwad method of decoder.

        Args:
            p (tensor): sampling locations, N*3
            npc (NerualPointCloud): shared npc object
            npc_col_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            pts_num (int, optional): sampled pts num along each ray. Defaults to N_surface.
            is_tracker (bool, optional): whether is called by tracker.
            cloud_pos (tensor, optional): point cloud position, used when called by tracker to re-calculate D. 
            pts_views_d (tensor): viweing directions
            dynamic_r_query (tensor, optional): if enabled dynamic radius, query radius for every pixel will be different.
            exposure_feat (tensor): exposure feature vector. Needs to be the same for all points in the batch.

        Returns:
            predicted colors for points p
        """
        c, _ = self.get_feature_at_pos(
            npc, p, npc_col_feats, is_tracker, cloud_pos, dynamic_r_query=dynamic_r_query)
        p = p.float().reshape(1, -1, 3)

        embedded_pts = self.embedder(p)
        embedded_input = embedded_pts

        if self.use_view_direction:
            pts_views_d = F.normalize(pts_views_d, p=2, dim=1)
            embedded_views_d = self.embedder_view_direction(pts_views_d)
            embedded_input = torch.cat(
                [embedded_pts, embedded_views_d], -1)
        h = embedded_input
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.actvn(h)
            if self.c_dim != 0:
                # hidden dim + (feature dim->hidden dim) -> hidden dim
                h = h + self.fc_c[i](c)
                # so for hidden layers in the decoder, its input comes from both its feature and embedded location.
            if i in self.skips:
                h = torch.cat([embedded_input, h], -1)
        out = self.output_linear(h)
        if self.encode_exposure:
            if exposure_feat is not None:
                affine_tensor = self.mlp_exposure(exposure_feat)
                rot, trans = affine_tensor[:9].reshape(
                    3, 3), affine_tensor[-3:]
                out = torch.matmul(out, rot) + trans
                out = torch.sigmoid(out)
            else:
                # apply exposure compensation outside "self.renderer.render_batch_ray" call in mapper
                # this is done when multiple exposure feature vectors are needed for different rays
                # during mapping. Each keyframe has its own exposure feature vector, while the forward
                # function of the MLP_color class assumes that all rays have the same exposure feature
                # vector.
                return out
        else:
            out = torch.sigmoid(out)

        return out


class POINT(nn.Module):
    """    
    Decoder for point represented features.

    Args:
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of decoder network
        pos_embedding_method (str): positional embedding method.
        use_view_direction (bool): use view direction or not.
    """

    def __init__(self, cfg, c_dim=32,
                 hidden_size=128,
                 pos_embedding_method='fourier', use_view_direction=False):
        super().__init__()

        self.geo_decoder = MLP_geometry(cfg=cfg, c_dim=c_dim,
                                        skips=[2], n_blocks=5, hidden_size=32,
                                        pos_embedding_method=pos_embedding_method)
        self.color_decoder = MLP_color(cfg=cfg, c_dim=c_dim,
                                       skips=[2], n_blocks=5, hidden_size=hidden_size,
                                       pos_embedding_method=pos_embedding_method,
                                       use_view_direction=use_view_direction)

    def forward(self, p, npc, stage, npc_geo_feats, npc_col_feats, pts_num=16, is_tracker=False, cloud_pos=None,
                pts_views_d=None, dynamic_r_query=None, exposure_feat=None):
        """
            Output occupancy/color and associated masks for validity

        Args:
            p (tensor): point locations
            npc (tensor): NeuralPointCloud object.
            stage (str): listed below.
            npc_geo_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            npc_col_feats (tensor): cloned from npc. Contains the optimizable parameters during mapping
            pts_num (int): number of points in sampled in each ray, used only by geo_decoder.
            is_tracker (bool): whether called by tracker.
            cloud_pos (tensor): (N,3)
            pts_views_d (tensor): used if color decoder encodes viewing directions.
            dynamic_r_query (tensor): (N,), used if dynamic radius enabled.
            exposure_feat (tensor): exposure feature vector. Needs to be the same for all points in the batch.

        Returns:
            raw (tensor): predicted color and occupancies for the points p
            ray_mask (tensor): boolen tensor. True if at least half of all points along the ray have neighbors
            point_mask (tensor): boolean tensor. False if at least two neighbors were not found for the point in question
        """
        device = f'cuda:{p.get_device()}'
        match stage:
            case 'geometry':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query)
                raw = torch.zeros(
                    geo_occ.shape[0], 4, device=device, dtype=torch.float)
                raw[..., -1] = geo_occ
                return raw, ray_mask, point_mask
            case 'color':
                geo_occ, ray_mask, point_mask = self.geo_decoder(p, npc, npc_geo_feats,
                                                                 pts_num=pts_num, is_tracker=is_tracker, cloud_pos=cloud_pos,
                                                                 dynamic_r_query=dynamic_r_query)
                raw = self.color_decoder(p, npc, npc_col_feats,                                # returned (N,4)
                                         is_tracker=is_tracker, cloud_pos=cloud_pos,
                                         pts_views_d=pts_views_d,
                                         dynamic_r_query=dynamic_r_query, exposure_feat=exposure_feat)
                raw = torch.cat([raw, geo_occ.unsqueeze(-1)], dim=-1)
                return raw, ray_mask, point_mask
