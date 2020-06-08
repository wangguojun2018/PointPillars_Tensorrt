#coding=utf-8
import os
from pathlib import Path
import torch
from google.protobuf import text_format
from second.builder import voxel_builder
from second.protos import pipeline_pb2
from second.builder import (box_coder_builder,second_builder)
from torch2trt import torch2trt
import argparse
import numpy as np


def generate_tensor_list(max_voxel_num,device,float_type):
    pillar_x = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_y = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_z = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    pillar_i = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    num_points_per_pillar = torch.ones([1, max_voxel_num], dtype=float_type, device=device)
    x_sub_shaped = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    y_sub_shaped = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    mask = torch.ones([1, 1, max_voxel_num, 100], dtype=float_type, device=device)
    example_list = [pillar_x, pillar_y, pillar_z, pillar_i,
                    num_points_per_pillar,x_sub_shaped, y_sub_shaped, mask]
    return example_list


def build_network(model_cfg):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    net = second_builder.build(
        model_cfg, voxel_generator, box_coder)
    return net


def inference(model_dir=None,filter_bg=False):


    model_dir = Path(model_dir)
    config_path='./configs/xyres_28_huituo.config'
    device = torch.device("cuda")
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    voxel_cfg=model_cfg.voxel_generator
    max_voxel_num=12000
    if filter_bg:
        max_voxel_num=1000
    net = build_network(model_cfg).to(device)

    #读取权重文件
    state_dict=torch.load(str(model_dir/'voxelnet.tckpt'))
    for key in ["global_step", "rpn_acc.total", "rpn_acc.count", "rpn_precision.total",
                "rpn_precision.count", "rpn_recall.total", "rpn_recall.count",
                "rpn_metrics.prec_total", "rpn_metrics.prec_count", "rpn_metrics.rec_total",
                "rpn_metrics.rec_count", "rpn_cls_loss.total", "rpn_cls_loss.count", "rpn_loc_loss.total",
                "rpn_loc_loss.count", "rpn_total_loss.total", "rpn_total_loss.count"]:
        if key in state_dict.keys():
            state_dict.pop(key)
    net.load_state_dict(state_dict)
    net.eval()

    #tensorrt引擎路径
    pfe_trt_path=str(model_dir/"pfe.trt")
    rpn_trt_path = str(model_dir / "rpn.trt")

    #生成模型虚假输入数据用于编译tensorrt引擎
    example_tensor=generate_tensor_list(max_voxel_num,float_type=torch.float32,device=device)

    #编译pillar feature net子网络引擎
    print("开始转换pfe子网络......")
    pfe_trt = torch2trt(net.voxel_feature_extractor, example_tensor, fp16_mode=True,
                        max_workspace_size=1 << 20)
    torch.save(pfe_trt.state_dict(), pfe_trt_path)

    # 编译rpn子网络引擎
    print("开始转换rpn子网络......")
    pc_range=np.array(voxel_cfg.point_cloud_range)
    vs=np.array(voxel_cfg.voxel_size)
    fp_size=((pc_range[3:]-pc_range[:3])/vs)[::-1].astype(np.int)
    rpn_input = torch.ones((1, 64, fp_size[1], fp_size[2]), dtype=torch.float32, device=device)
    rpn_trt = torch2trt(net.rpn, [rpn_input], fp16_mode=True, max_workspace_size=1 << 20)
    torch.save(rpn_trt.state_dict(), rpn_trt_path)

    print("export trt model successful")





if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_dir",type=str,default="./weights/pointpillars_roadside_28_090609271113",help="指定模型权重目录")
    parse.add_argument("--bg_filter",action='store_true',help="是否集成背景滤波")
    parse.set_defaults(bg_filter=False)
    args=parse.parse_args()
    print("集成背景滤波{}".format(args.bg_filter))
    print("模型目录 {}".format(args.model_dir))
    inference(model_dir=args.model_dir,filter_bg=args.bg_filter)