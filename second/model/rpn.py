import time

import numpy as np
import torch
from torch import nn
from torchplus.nn import Sequential,GroupNorm
from second.utils.tools import change_default_args
class RPN(nn.Module):
    def __init__(self,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._box_code_size=box_code_size
        self._num_class=num_class
        self._num_direction_bins=num_direction_bins
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides=[int(i) for i in upsample_strides]

        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_groupnorm:
            BatchNorm2d = change_default_args(
                num_groups=num_groups, eps=1e-3)(GroupNorm)
        else:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(num_input_features, num_filters[0], 3,
                stride=layer_strides[0],bias=False),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),)
        for i in range(layer_nums[0]):
            self.block1.add(
                nn.Conv2d(num_filters[0], num_filters[0], 3,padding=1,bias=False))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            nn.ConvTranspose2d(num_filters[0],num_upsample_filters[0],
                upsample_strides[0],stride=upsample_strides[0],bias=False),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),)
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(block2_input_filters,num_filters[1],3,
                stride=layer_strides[1],bias=False),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),)
        for i in range(layer_nums[1]):
            self.block2.add(
                nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1,bias=False))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            nn.ConvTranspose2d(num_filters[1],num_upsample_filters[1],
                upsample_strides[1],stride=upsample_strides[1],bias=False),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),)
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2],bias=False),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),)
        for i in range(layer_nums[2]):
            self.block3.add(nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1,bias=False))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            nn.ConvTranspose2d(
                num_filters[2],num_upsample_filters[2],
                upsample_strides[2],stride=upsample_strides[2],bias=False),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),)

        num_cls = num_anchor_per_loc * num_class
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters),num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)  #(1,14,H,W)
        cls_preds = self.conv_cls(x)  #(1,4,H,W)
        dir_cls_preds = self.conv_dir_cls(x)  ##(1,4,H,W)
        C, H, W = box_preds.shape[1:]
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,
                                   self._box_code_size, H, W).permute(
            0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,
                                   self._num_class, H, W).permute(
            0, 1, 3, 4, 2).contiguous()
        dir_cls_preds = dir_cls_preds.view(-1, self._num_anchor_per_loc, self._num_direction_bins, H,
                W).permute(0, 1, 3, 4, 2).contiguous()
        ret_tuple =(box_preds, cls_preds,dir_cls_preds)
        return ret_tuple
