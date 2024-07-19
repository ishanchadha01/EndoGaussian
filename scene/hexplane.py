import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# import tinycudann as tcnn


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs

def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    
    for scale_id,  grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class HexPlaneField(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)
        # print(f"Initialized model grids: {self.grids}")
        print("feature_dim:",self.feat_dim)

    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ])
        self.aabb = nn.Parameter(aabb,requires_grad=True) # !!!!!
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)
        
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
            
        return features

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        features = self.get_density(pts, timestamps)

        return features

# kplanes_config = {
#     'grid_dimensions': 2,
#     'input_coordinate_dim': 4,
#     'output_coordinate_dim': 64,
#     'resolution': [64, 64, 64, 100]

# },
# multires = [1, 2, 4, 8],
# defor_depth = 0,
# net_width = 32,
# plane_tv_weight = 0,
# time_smoothness_weight = 0,
# l1_time_planes =  0,
# weight_decay_iteration=0,

# hash_params = {
# 	"otype": "Grid",           # Component type.
# 	"type": "Hash",            # Type of backing storage of the
# 	                           # grids. Can be "Hash", "Tiled"
# 	                           # or "Dense".
# 	"n_levels": 16,            # Number of levels (resolutions)
# 	"n_features_per_level": 2, # Dimensionality of feature vector
# 	                           # stored in each level's entries.
# 	"log2_hashmap_size": 19,   # If type is "Hash", is the base-2
# 	                           # logarithm of the number of elements
# 	                           # in each backing hash table.
# 	"base_resolution": 16,     # The resolution of the coarsest le-
# 	                           # vel is base_resolution^input_dims.
# 	"per_level_scale": 2.0,    # The geometric growth factor, i.e.
# 	                           # the factor by which the resolution
# 	                           # of each grid is larger (per axis)
# 	                           # than that of the preceding level.
# 	"interpolation": "Linear"  # How to interpolate nearby grid
# 	                           # lookups. Can be "Nearest", "Linear",
# 	                           # or "Smoothstep" (for smooth deri-
# 	                           # vatives).
# }
# # number of encoded dimensions is n_levels * n_features_per_level


# class HashHexPlaneField(nn.Module):
#     def __init__(self, bounds, kplanes_config, multires):
#         super().__init__()
#         aabb = torch.tensor([[bounds,bounds,bounds],
#                              [-bounds,-bounds,-bounds]])
#         self.aabb = nn.Parameter(aabb, requires_grad=False)
        
#         self.grid_config =  [kplanes_config] 
#         self.multiscale_res_multipliers = multires
#         self.concat_features = True

#         # Init planes
#         self.grids = None
#         self.build_encoding()
#         self.feat_dim = 16 * 8 # num levels * num features per level, multiply along 6 grids, figure out way to make 8 into 64
#         print(f"feature_dim: {self.feat_dim}")


#     def build_encoding(self):
#         self.grids = nn.ModuleList()
#         #TODO: if this doesnt work, then make only spatial grids multires
#         for _ in range(6):
#             config = dict(
#                 otype="HashGrid",
#                 n_levels=16, # original multires has 4 levels
#                 n_features_per_level=8, # this is 2 for same reason as below
#                 log2_hashmap_size=22, # not sure how big this should be, bigger prolly better tho
#                 base_resolution=2 ** 5, # original multires is 4 levels 64,64,64,100, so trying 32,64,128,256
#                 per_level_scale=2.0, # we want to go up by powers of 2 between resolutions for now
#             )
#             # we're passing in xt, yt, zt, xy, yz, xz, so 2 inputs per grid
#             grid = tcnn.Encoding(2, config)
#             self.grids.append(grid)

#         # # for time dont do multires
#         # for _ in range(3):
#         #     config = dict(
#         #         otype="HashGrid",
#         #         n_levels=4, # original multires has 4 levels
#         #         n_features_per_level=8, # this is 2 for same reason as below
#         #         log2_hashmap_size=22, # not sure how big this should be, bigger prolly better tho
#         #         base_resolution=2 ** 5, # original multires is 4 levels 64,64,64,100, so trying 32,64,128,256
#         #         per_level_scale=2.0, # we want to go up by powers of 2 between resolutions for now
#         #     )
#         #     # we're passing in xt, yt, zt, xy, yz, xz, so 2 inputs per grid
#         #     grid = tcnn.Encoding(2, config)
#         #     self.grids.append(grid)


#     def set_aabb(self, xyz_max, xyz_min):
#         aabb = torch.tensor([
#             xyz_max,
#             xyz_min
#         ])
#         self.aabb = nn.Parameter(aabb,requires_grad=True) # !!!!!
#         print("Voxel Plane: set aabb=",self.aabb)

#     def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
#         """Computes and returns the densities."""

#         import pdb
#         pdb.set_trace()
#         pts = normalize_aabb(pts, self.aabb)
#         pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
#         pts = pts.reshape(-1, pts.shape[-1])
        
#         # get pairings
#         # xt = pts[..., [0,3]]
#         # yt = pts[..., [1,3]]
#         # zt = pts[..., [2,3]]
#         # xy = pts[..., [0,1]]
#         # yz = pts[..., [1,2]]
#         # xz = pts[..., [1,3]]

#         # # add axis for num batches
#         # if pts.dim() == 2:
#         #     coords = coords.unsqueeze(0)
#         coord_combs = list(itertools.combinations(
#             range(pts.shape[-1]), 2)
#         )
#         feats = 1.
#         for combo, enc in zip(coord_combs, self.grids): #TODO: should only need 1 grid if they all the same anyways
#             tcnn_input = pts[...,combo].view(-1, 2)
#             tcnn_output = enc(tcnn_input)
#             pts_enc = tcnn_output.view(*pts.shape[:-1], tcnn_output.shape[-1])
#             feats = feats * pts_enc
#             # this is placeholder for now, i really need to compute product of each all 6 planes
#             # for each resolution it seems. how does this compute factored product of low rank matrices tho?
#             # seems to be formulated differently than original hexplane strategy
#             #TODO: are features interpolated using this hashing strategy?

#         # if len(feats) < 1:
#         #     feats = torch.zeros((0, 1)).to(feats.device)
            
#         return feats

#     def forward(self,
#                 pts: torch.Tensor,
#                 timestamps: Optional[torch.Tensor] = None):
#         features = self.get_density(pts, timestamps)

#         return features