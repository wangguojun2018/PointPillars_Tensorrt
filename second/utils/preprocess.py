import numpy as np

import torch


def voxel_padding(voxels, num_points, coords,max_voxel_num):
    num_points_per_voxel=voxels.shape[1]
    voxel_num=voxels.shape[0]
    voxel_mask = np.zeros(max_voxel_num,dtype=np.int8)
    assert voxel_num<=max_voxel_num,"voxel num greater than max voxel num"
    pad_num=max_voxel_num-voxel_num
    voxel_mask[:voxel_num]=1
    voxels_pad=np.zeros((pad_num,num_points_per_voxel,4),dtype=voxels.dtype)
    num_points_pad=np.ones(pad_num,dtype=num_points.dtype)
    coords_pad=np.zeros((pad_num,3),dtype=coords.dtype)
    voxels=np.concatenate([voxels,voxels_pad],axis=0)
    num_points=np.concatenate([num_points,num_points_pad],axis=0)
    coords=np.concatenate([coords,coords_pad],axis=0)
    return voxels,num_points,coords,voxel_mask


def get_paddings_indicator_np(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = np.expand_dims(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = np.arange(
        max_num, dtype=np.int).reshape(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.astype(np.int32) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

def example_to_tensorlist(example, device=None,float_type=torch.float32):
    example_list = [None] * 13
    pillar_x = example['voxels'][:, :, :, 0][:, np.newaxis, :, :]  # (N,1,K,T)
    pillar_y = example['voxels'][:, :, :, 1][:, np.newaxis, :, :]
    pillar_z = example['voxels'][:, :, :, 2][:, np.newaxis, :, :]
    pillar_i = example['voxels'][:, :, :, 3][:, np.newaxis, :, :]
    num_points_per_pillar = example['num_points']  # (N,K,)
    coors = example['coordinates']  # (N,K,3)
    anchors = example['anchors']  # (B,num_anchors,7)
    image_ids = [int(elem['image_idx']) for elem in example['metadata']]
    image_ids = np.array(image_ids, dtype=np.int32)
    voxel_mask = example['voxel_mask']  # (N,K)
    # ################################################################
    # Find distance of x, y, z from pillar center
    coors_x = example['coordinates'][:, :, 2]  # (N,K)
    coors_y = example['coordinates'][:, :, 1]

    x_sub = coors_x[:, np.newaxis, :, np.newaxis] * 0.28 - 29.7  # Pillars的中心的位置坐标 (N,1,K,1)
    y_sub = coors_y[:, np.newaxis, :, np.newaxis] * 0.28 - 66.18
    # print("before repeat x_sub nan is ",torch.nonzero(torch.isnan(x_sub)).shape)
    # print("before repeat y_sub nan is ", torch.nonzero(torch.isnan(y_sub)).shape)

    x_sub_shaped = x_sub.repeat(pillar_x.shape[3], -1)
    y_sub_shaped = y_sub.repeat(pillar_x.shape[3], -1)  # (N,1,K,T)
    # print("after repeat x_sub nan is ", torch.nonzero(torch.isnan(x_sub_shaped)).shape)
    # print("after repeat y_sub nan is ", torch.nonzero(torch.isnan(y_sub_shaped)).shape)
    num_points_for_a_pillar = pillar_x.shape[3]  # (T)
    mask = get_paddings_indicator_np(num_points_per_pillar, num_points_for_a_pillar, axis=0)  # (N,T,K)
    mask = mask.transpose(0, 2, 1)  # (N,K,T)
    mask = mask[:, np.newaxis, :, :]  # (N,1,K,T)
    mask = mask.astype(pillar_x.dtype)

    example_list[0] = torch.tensor(pillar_x, dtype=float_type, device=device)
    example_list[1] = torch.tensor(pillar_y, dtype=float_type, device=device)
    example_list[2] = torch.tensor(pillar_z, dtype=float_type, device=device)
    example_list[3] = torch.tensor(pillar_i, dtype=float_type, device=device)
    example_list[4] = torch.tensor(num_points_per_pillar, dtype=float_type, device=device)
    example_list[5] = torch.tensor(x_sub_shaped, dtype=float_type, device=device)
    example_list[6] = torch.tensor(y_sub_shaped, dtype=float_type, device=device)
    example_list[7] = torch.tensor(mask, dtype=float_type, device=device)
    example_list[8] = torch.tensor(coors, dtype=torch.int32, device=device)
    example_list[9] = torch.tensor(voxel_mask, dtype=torch.bool, device=device)
    example_list[10] = torch.tensor(anchors, dtype=float_type, device=device)
    example_list[11] = torch.tensor(image_ids, dtype=torch.int32, device=device)
    if 'anchors_mask' in example.keys():
        example_list[12]=torch.tensor(example['anchors_mask'], dtype=torch.bool, device=device)
    else:
        example_list[12]=None
    return example_list



def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch
