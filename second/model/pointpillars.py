"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False,use_groupnorm=False,
                 num_groups=32):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels=in_channels
        # self.linear = nn.Linear(in_channels, self.units,bias=False)
        # self.norm = nn.BatchNorm1d(self.units,eps=1e-3,momentum=0.01)

        self.norm1 = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)
        self.norm2 = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        if use_groupnorm:
            self.norm1=nn.GroupNorm(num_groups,self.units)
            self.norm2=nn.GroupNorm(num_groups,self.units)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1,bias=False)
        # self.conv2 = nn.Conv2d(in_channels=self.units, out_channels=self.units, kernel_size=(1, 34), stride=(1, 1), dilation=(1, 3))
        self.conv2=nn.Conv2d(in_channels=self.units,out_channels=self.units,kernel_size=(1,100),stride=1,bias=False)

    def forward(self, inputs):
        x=self.conv1(inputs)
        x=self.norm1(x)
        x=F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        # x = self.conv2(x)  #(N,64,K,1)
        return x
        # if inputs.shape[0]>65000:
        #     inputs1,inputs2,inputs3=torch.chunk(inputs,3,dim=0)
        #     x1=self.linear(inputs1)
        #     x2 = self.linear(inputs2)
        #     x3 = self.linear(inputs3)
        #     x=torch.cat([x1,x2,x3],dim=0)
        # else:
        #     inputs=inputs.permute(0,2,3,1)
        #     x=self.linear(inputs)
        #     x = self.norm(x.permute(0,3,1,2).contiguous()).permute(0, 2,3,
        #                                                            1).contiguous()
        #     x = F.relu(x)
        #
        #     x_max = torch.max(x, dim=2, keepdim=True)[0]
        #
        #     if self.last_vfe:
        #         return x_max   #
        #     else:
        #         x_repeat = x_max.repeat(1,1, inputs.shape[3], 1)
        #         x_concatenated = torch.cat([x, x_repeat], dim=3)
        #         return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_groupnorm=False,
                 num_groups=32,
                 num_filters=(64, ),
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, last_layer=last_layer,
                    use_groupnorm=use_groupnorm,num_groups=num_groups))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    # def forward(self, features, num_voxels, coors):
    #     device = features.device
    #
    #     dtype = features.dtype
    #     # Find distance of x, y, and z from cluster center
    #     points_mean = features[:, :, :3].sum(
    #         dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
    #     f_cluster = features[:, :, :3] - points_mean
    #
    #     # Find distance of x, y, and z from pillar center
    #     f_center = torch.zeros_like(features[:, :, :2])
    #     f_center[:, :, 0] = features[:, :, 0] - (
    #         coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
    #     f_center[:, :, 1] = features[:, :, 1] - (
    #         coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
    #
    #     # Combine together feature decorations
    #     features_ls = [features, f_cluster, f_center]
    #     if self._with_distance:
    #         points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
    #         features_ls.append(points_dist)
    #     features = torch.cat(features_ls, dim=-1)
    #     # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
    #     # empty pillars remain set to zeros.
    #     voxel_count = features.shape[1]
    #     mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
    #     mask = torch.unsqueeze(mask, -1).type_as(features)
    #     features *= mask
    #
    #     # Forward pass through PFNLayers
    #     for pfn in self.pfn_layers:
    #         features = pfn(features)
    #
    #     return features.squeeze()
    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_voxels, x_sub_shaped, y_sub_shaped, mask):

        # Find distance of x, y, and z from cluster center,(N,1,K,T)
        #num_voxels:(N,K)
        #x_sub_shaped:#(N,1,K,T)
        #mask:##(N,1,K,T)
        # pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 3)

        pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 1)  #(N,3,K,T)

        N,K=num_voxels.shape
        points_mean = pillar_xyz.sum(dim=3, keepdim=True) / num_voxels.view(N,1,K,1) #(N,3,K,T)
        f_cluster = pillar_xyz - points_mean

        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped

        f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 1) #(N,2,K,T)

        pillar_xyzi = torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 1)
        features_list = [pillar_xyzi, f_cluster, f_center_concat]


        features = torch.cat(features_list, dim=1)  #(N,9,K,T)
        masked_features = features * mask  #

        pillar_features = self.pfn_layers[0](masked_features) #(1,64,K,1)
        return pillar_features

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords,voxel_mask, batch_size):

        # batch_canvas will be the final output.
        # voxel_features=voxel_features[:voxel_num]
        # coords=coords[:voxel_num]
        #voxel_features:(B,K,64)
        #coords:(N,K,3)
        #voxel_mask:(N,K)
        batch_canvas = []

        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            # batch_mask = coords[:, 0] == batch_itt
            # batch_mask=batch_mask*voxel_mask
            # this_coords = coords[batch_mask, :]
            this_coords=coords[batch_itt][voxel_mask[batch_itt]]

            indices = this_coords[:, 1] * self.nx + this_coords[:, 2]
            indices = indices.type(torch.float)
            voxels = voxel_features[batch_itt][voxel_mask[batch_itt]]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            # canvas[:, indices] = voxels

            indices_2d = indices.view(1, -1)
            ones = torch.ones([self.nchannels, 1], dtype=torch.float, device=voxel_features.device)
            indices_num_channel = torch.mm(ones, indices_2d)
            indices_num_channel = indices_num_channel.type(torch.int64)
            canvas.scatter_(1, indices_num_channel, voxels)

            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas
