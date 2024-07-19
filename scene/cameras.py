#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, mask, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 data_device = "cuda", time = 0, Znear=None, Zfar=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.mask = mask
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width)).cuda()
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        use_brdf_est = True #TODO put this in a config
        if use_brdf_est:
            self.preprocess_img_le()

        
        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            # ENDONERF
            self.zfar = 120.0
            self.znear = 0.01
            # SCARED
            # self.zfar = 300
            # self.znear= 0.03
            
        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def preprocess_img_le(self):
        """
        Undistort images according to lightneus, second pass
        """

        channels, rows, cols = self.original_image.shape

        #TODO: put below magic nums from calibration in conf
        g= 2.0 # Autogain unknown for ct1a, estimates from 1 to 3
        gamma = 2.2 # gamma correction, generally constant
        k = 2.5 # decay power from emitted light
        f = 767.45 # average of fx and fy, TODO compute differently in different directions
        h = 1080
        w = 1350
        # (252,0) pixel coord varies between 160<alpha<170
        p0 = np.array([252,0]) # point at corner of fov
        pcenter = np.array([(h-1)/2, (w-1)/2])
        centered_p0 = p0 - pcenter
        fov = 165/2 # varies between 160 and 170 for this c3vd endoscope
        alpha = fov/2
        d = f * np.tan(alpha)
        px_size = np.array([d*np.sin(alpha), d*np.cos(alpha)]) / centered_p0 # get pixel sizes, row/col correspond to y/x

        # Compute alpha and then Le for every point based on pixel size
        rows, cols = np.meshgrid(np.arange(w), np.arange(h))
        row_dist = np.abs(rows - pcenter[0])
        col_dist = np.abs(cols - pcenter[1])
        dists = np.stack((row_dist, col_dist), axis=-1) * np.expand_dims(px_size, axis=(0,1))
        dists = np.linalg.norm(dists, axis=2)
        alpha = np.arctan2(dists, f)
        Le = np.power(np.cos(alpha), k)

        # Compute normalized image
        Le = torch.from_numpy(Le[np.newaxis, :,:])
        self.original_image.pow_(gamma).div_(Le * g)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

