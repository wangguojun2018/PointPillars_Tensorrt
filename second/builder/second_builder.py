# Copyright 2017 yanyan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""VoxelNet builder.
"""

from second.protos import second_pb2
from second.model.voxelnet import VoxelNet
import numpy as np

def get_downsample_factor(model_config):
    downsample_factor = np.prod(model_config.rpn.layer_strides)
    if len(model_config.rpn.upsample_strides) > 0:
        downsample_factor /= model_config.rpn.upsample_strides[-1]
    downsample_factor *= model_config.middle_feature_extractor.downsample_factor
    downsample_factor = np.round(downsample_factor).astype(np.int64)
    assert downsample_factor > 0
    return downsample_factor

def build(model_cfg: second_pb2.VoxelNet, voxel_generator,
          box_coder,):
    """build second pytorch instance.
    """
    if not isinstance(model_cfg, second_pb2.VoxelNet):
        raise ValueError('model_cfg not of type ' 'second_pb2.VoxelNet.')
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    grid_size = voxel_generator.grid_size
    dense_shape = [1] + grid_size[::-1].tolist() + [vfe_num_filters[-1]]
    classes_cfg = model_cfg.target_assigner.class_settings
    num_class = len(classes_cfg)+1

    nms_pre_max_sizes = [c.nms_pre_max_size for c in classes_cfg]
    nms_post_max_sizes = [c.nms_post_max_size for c in classes_cfg]
    nms_score_thresholds = [c.nms_score_threshold for c in classes_cfg]
    nms_iou_thresholds = [c.nms_iou_threshold for c in classes_cfg]
    num_input_features = model_cfg.num_point_features

    net = VoxelNet(
        output_shape=dense_shape,
        num_class=num_class,
        vfe_num_filters=vfe_num_filters,
        middle_num_input_features=model_cfg.middle_feature_extractor.num_input_features,
        rpn_num_input_features=model_cfg.rpn.num_input_features,
        rpn_layer_nums=list(model_cfg.rpn.layer_nums),
        rpn_layer_strides=list(model_cfg.rpn.layer_strides),
        rpn_num_filters=list(model_cfg.rpn.num_filters),
        rpn_upsample_strides=list(model_cfg.rpn.upsample_strides),
        rpn_num_upsample_filters=list(model_cfg.rpn.num_upsample_filters),
        nms_score_thresholds=nms_score_thresholds,
        nms_pre_max_sizes=nms_pre_max_sizes,
        nms_post_max_sizes=nms_post_max_sizes,
        nms_iou_thresholds=nms_iou_thresholds,
        box_coder=box_coder,
        num_input_features=num_input_features,
        num_groups=model_cfg.rpn.num_groups,
        use_groupnorm=model_cfg.rpn.use_groupnorm,
        voxel_generator=voxel_generator,
        post_center_range=list(model_cfg.post_center_limit_range),
        dir_offset=model_cfg.direction_offset, #0.0
        num_direction_bins=model_cfg.num_direction_bins, #2
    )
    return net
