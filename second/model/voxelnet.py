import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from second.core import box_torch_ops

from second.model import pointpillars, rpn

class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_num_filters=[32, 128],
                 middle_num_input_features=-1,
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_groupnorm=False,
                 num_groups=32,
                 nms_score_thresholds=None,
                 nms_pre_max_sizes=None,
                 nms_post_max_sizes=None,
                 nms_iou_thresholds=None,
                 box_coder=None,
                 voxel_generator=None,
                 post_center_range=None,
                 dir_offset=0.0,
                 num_direction_bins=2,
                 name='voxelnet'):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._nms_score_thresholds = nms_score_thresholds
        self._nms_pre_max_sizes = nms_pre_max_sizes
        self._nms_post_max_sizes = nms_post_max_sizes
        self._nms_iou_thresholds = nms_iou_thresholds
        self._num_input_features = num_input_features
        self._box_coder = box_coder
        self.voxel_generator = voxel_generator
        self._dir_offset = dir_offset
        self._post_center_range = post_center_range or []
        self._num_direction_bins = num_direction_bins

        self.voxel_feature_extractor = pointpillars.PillarFeatureNet(
            num_input_features,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            num_filters=vfe_num_filters,
            voxel_size=self.voxel_generator.voxel_size,
            pc_range=self.voxel_generator.point_cloud_range,
        )

        self.middle_feature_extractor = pointpillars.PointPillarsScatter(
            output_shape,num_input_features=middle_num_input_features,)


        self.rpn = rpn.RPN(
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_features=rpn_num_input_features,
            num_anchor_per_loc=2,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=self._box_coder.code_size,
            num_direction_bins=self._num_direction_bins)


    def forward(self, pillar_x,pillar_y,pillar_z,pillar_i,
                num_points,x_sub_shaped,y_sub_shaped,mask,
                coors,voxel_mask,batch_anchors,image_ids=None,
                anchors_mask=None):
        # training input [0:pillar_x, 1:pillar_y, 2:pillar_z, 3:pillar_i,
        #                 4:num_points_per_pillar, 5:x_sub_shaped, 6:y_sub_shaped, 7:mask, 8:coors,9:voxel_num
        #                 10:anchors,11:image_ids, 12:labels, 13:reg_targets]

        # batch_anchors=example[9]
        batch_size=batch_anchors.shape[0]
        pillar_features = self.voxel_feature_extractor(pillar_x, pillar_y, pillar_z, pillar_i,
                                                      num_points, x_sub_shaped, y_sub_shaped, mask)
        pillar_features = pillar_features.squeeze(-1)
        pillar_features = pillar_features.permute(0,2, 1)  # (N,K,64)
        spatial_features = self.middle_feature_extractor(
            pillar_features, coors,voxel_mask, batch_size)
        preds_tuple = self.rpn(spatial_features)

        box_preds = preds_tuple[0].view(batch_size, -1, self._box_coder.code_size)
        err_msg = f"num_anchors={batch_anchors.shape[1]}, but num_output={box_preds.shape[1]}. please check size"
        assert batch_anchors.shape[1] == box_preds.shape[1], err_msg

        res = self.predict(batch_anchors,preds_tuple,image_ids=image_ids,anchors_mask=anchors_mask)

        return res

    def predict(self, anchors,preds_tuple,image_ids=None,anchors_mask=None):
        """start with v1.6.0, this function don't contain any kitti-specific code.
        Returns:
            predict: list of pred_tuple.
            pred_tuple: {
                box3d_lidar: [N, 7] 3d box.
                scores: [N]
                label_preds: [N]
                image_id:[N]
            }
        """
        batch_size = anchors.shape[0]
        meta_list = image_ids
        batch_anchors = anchors.view(batch_size, -1,anchors.shape[-1])
        batch_anchors_mask = [None] * batch_size
        if anchors_mask is not None:
            batch_anchors_mask = anchors_mask.view(batch_size, -1)

        batch_box_preds = preds_tuple[0]
        batch_cls_preds = preds_tuple[1]
        batch__dir_preds=preds_tuple[2]

        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               self._num_class)


        batch_dir_preds = batch__dir_preds.view(batch_size, -1,
                                              self._num_direction_bins)



        predictions_tuples = []

        post_center_range = torch.tensor(
            self._post_center_range,
            dtype=batch_box_preds.dtype,
            device=batch_box_preds.device).float()
        for box_preds, cls_preds, dir_preds, meta,a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, meta_list,batch_anchors_mask):

            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                dir_preds = dir_preds[a_mask]

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            dir_labels = torch.max(dir_preds, dim=-1)[1]

            top_scores = F.softmax(cls_preds, dim=-1)[...,1]
            top_labels=torch.zeros_like(top_scores,device=top_scores.device,dtype=torch.int32)

            if self._nms_score_thresholds[0] > 0.0:
                top_scores_keep = top_scores >= self._nms_score_thresholds[0]

                top_scores=top_scores[top_scores_keep]

                box_preds = box_preds[top_scores_keep]

                dir_labels = dir_labels[top_scores_keep]
                top_labels = top_labels[top_scores_keep]

            if top_scores.shape[0]>0:
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = box_torch_ops.nms(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_sizes[0],
                    post_max_size=self._nms_post_max_sizes[0],
                    iou_threshold=self._nms_iou_thresholds[0],)
                # if selected is not None:
                selected_boxes = box_preds[selected]
                selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
                if selected_boxes.shape[0] != 0:
                    box_preds = selected_boxes
                    scores = selected_scores
                    label_preds = selected_labels

                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / self._num_direction_bins)
                    dir_rot = box_torch_ops.limit_period(
                        box_preds[..., 6] - self._dir_offset,
                        0, period)
                    box_preds[...,6] = dir_rot + self._dir_offset + period * dir_labels.to(
                            box_preds.dtype)
                    final_box_preds = box_preds
                    final_scores = scores
                    final_labels = label_preds

                   #post roi process
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_tuple = (
                        final_box_preds[mask],
                        final_scores[mask],
                        final_labels[mask],
                        meta, )
                else:
                    dtype = batch_box_preds.dtype
                    device = batch_box_preds.device
                    predictions_tuple = (
                        torch.zeros([0, box_preds.shape[-1]],
                                    dtype=dtype,
                                    device=device),
                        torch.zeros([0], dtype=dtype, device=device),
                        torch.zeros([0], dtype=top_labels.dtype, device=device),
                        meta,)
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_tuple = (
                    torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    torch.zeros([0], dtype=dtype, device=device),
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    meta,)
            predictions_tuples.append(predictions_tuple)
        return predictions_tuples



