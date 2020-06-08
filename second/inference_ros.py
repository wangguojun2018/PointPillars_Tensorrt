#coding=utf-8
from second.utils.bg_filter import bg_filter
import os
import time
import numpy as np
import torch
from google.protobuf import text_format
from second.builder import anchor_generator_builder, voxel_builder

from second.protos import pipeline_pb2
from second.builder import (box_coder_builder,
                                    second_builder)
from ros_numpy import point_cloud2
from second.utils.preprocess import example_to_tensorlist
from second.builder.second_builder import get_downsample_factor
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from numpy.lib.recfunctions import structured_to_unstructured
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
from visualization_msgs.msg import Marker,MarkerArray
from second.core import box_np_ops
import argparse
from torch2trt import TRTModule
from second.utils.preprocess import voxel_padding

# DUMMY_FIELD_PREFIX = '__'

class SecondROS:
    '''
    模型ros接口类，通过订阅rslidar_points点云话题，执行lidar_callback回调函数输出检测结果（边界框和得分）
    '''
    def __init__(self,model_dir,bg_filter=False,is_tensorrt=True,bg_dir=None,anchors_area=0):
        rospy.init_node('second_ros')
        if 'BG' not in model_dir:
            assert bg_filter is False,"非背景滤波模型无法开启背景滤波"
        else:
            assert bg_filter is True,"背景滤波模型必须开启背景滤波模块"
        print("使用tensorrt 引擎 {}, 使用背景滤波 {}".format(is_tensorrt,bg_filter))
        print("背景表目录为： {}".format(bg_dir))
        # Subscriber
        self.model = SecondModel(model_dir=model_dir,filter=bg_filter,
                                 tensorrt=is_tensorrt,bg_dir=bg_dir,anchors_area=anchors_area)
        #等待点云话题 /rslidar_points
        print("等待点云话题/rslidar_points......")
        self.sub_lidar = rospy.Subscriber("/rslidar_points", PointCloud2, self.lidar_callback, queue_size=1)

        # Publisher
        self.pub_bbox = rospy.Publisher("/boxes", BoundingBoxArray, queue_size=1)
        self.pub_text=rospy.Publisher("/scores",MarkerArray,queue_size=0)
        self.pub_cloud = rospy.Publisher("/cloud_filtered", PointCloud2, queue_size=0)

        rospy.spin()

    def lidar_callback(self, msg):
        pc_arr=point_cloud2.pointcloud2_to_array(msg)
        pc_arr = structured_to_unstructured(pc_arr)
        pc_arr=pc_arr.reshape(-1,4)
        lidar_boxes,lidar_scores = self.model.predcit(pc_arr)
        # points.dtype=[('x', np.float32),('y', np.float32),('z', np.float32),('intensity', np.float32)]
        # cloud_msg=point_cloud2.array_to_pointcloud2(points,rospy.Time.now(),"rslidar")

        num_detects = len(lidar_boxes)
        arr_bbox = BoundingBoxArray()
        arr_score=MarkerArray()
        for i in range(num_detects):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()

            bbox.pose.position.x = float(lidar_boxes[i][0])
            bbox.pose.position.y = float(lidar_boxes[i][1])
            bbox.pose.position.z = float(lidar_boxes[i][2])
            # bbox.pose.position.z = float(lidar_boxes[i][2]) + float(lidar_boxes[i][5]) / 2
            bbox.dimensions.x = float(lidar_boxes[i][3])  # width
            bbox.dimensions.y = float(lidar_boxes[i][4])  # length
            bbox.dimensions.z = float(lidar_boxes[i][5])  # height

            q = Quaternion(axis=(0, 0, 1), radians=float(-lidar_boxes[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w

            arr_bbox.boxes.append(bbox)

            marker = Marker()
            marker.header.frame_id =msg.header.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "basic_shapes"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.lifetime=rospy.Duration(0.15)
            marker.scale.x = 4
            marker.scale.y = 4
            marker.scale.z = 4

            # Marker的颜色和透明度
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1
            marker.color.a = 1
            marker.pose.position.x=float(lidar_boxes[i][0])
            marker.pose.position.y = float(lidar_boxes[i][1])
            marker.pose.position.z = float(lidar_boxes[i][2]) + float(lidar_boxes[i][5]) / 2
            marker.text=str(np.around(lidar_scores[i],2))
            arr_score.markers.append(marker)
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        print("Number of detections: {}".format(num_detects))

        self.pub_bbox.publish(arr_bbox)
        self.pub_text.publish(arr_score)
        # self.pub_cloud.publish(cloud_msg)



class SecondModel:
    def __init__(self, model_dir,filter=True,tensorrt=True,bg_dir=None,anchors_area=0):
        self.model_dir=model_dir
        self.config_path = os.path.join('./configs',"xyres_28_huituo.config")
        self.anchors_area=anchors_area
        self.bg_filter=None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)

        model_cfg = config.model.second
        classes_cfg= model_cfg.target_assigner.class_settings
        if filter:
            assert bg_dir is not None,"如果采用背景滤波，必须输入背景表目录"
            self.bg_filter = bg_filter(model_cfg.voxel_generator,num_path=bg_dir+'/num_table.txt',
                                       var_path=bg_dir+'/var_table.txt',num_point=2,diff=0.03,is_statistic=False)
        net = self.build_network(model_cfg).to(self.device)

        self.anchor_generator = anchor_generator_builder.build(classes_cfg[0])
        self.voxel_generator = net.voxel_generator
        self.voxel_size=self.voxel_generator.voxel_size
        self.grid_size=self.voxel_generator.grid_size
        self.pc_range = self.voxel_generator.point_cloud_range
        self.max_voxel_num=12000
        if filter:
            self.max_voxel_num=2000
        ckpt_path=os.path.join(model_dir,'voxelnet.tckpt')
        state_dict=torch.load(ckpt_path)
        for key in ["global_step", "rpn_acc.total", "rpn_acc.count", "rpn_precision.total",
                    "rpn_precision.count", "rpn_recall.total", "rpn_recall.count",
                    "rpn_metrics.prec_total", "rpn_metrics.prec_count", "rpn_metrics.rec_total",
                    "rpn_metrics.rec_count", "rpn_cls_loss.total", "rpn_cls_loss.count", "rpn_loc_loss.total",
                    "rpn_loc_loss.count", "rpn_total_loss.total", "rpn_total_loss.count"]:
            if key in state_dict.keys():
                state_dict.pop(key)
        net.load_state_dict(state_dict)

        if tensorrt:
            pfe_trt = TRTModule()
            pfe_trt.load_state_dict(torch.load(os.path.join(model_dir,'pfe.trt')))
            rpn_trt = TRTModule()
            rpn_trt.load_state_dict(torch.load(os.path.join(model_dir,'rpn.trt')))
            net.voxel_feature_extractor = pfe_trt
            net.rpn = rpn_trt

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // get_downsample_factor(model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]
        anchors = self.anchor_generator.generate_anchors(feature_map_size)

        self.anchors = anchors.reshape((1, -1, 7))
        self.anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        # self.anchors_bv=anchors_bv.reshape((1,-1,4))

        if tensorrt:
            self.float_dtype = torch.float32
        else:
            self.float_dtype = torch.float32
        self.net=net.eval()

    def build_network(self,model_cfg):
        voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
        box_coder = box_coder_builder.build(model_cfg.box_coder)
        net = second_builder.build(
            model_cfg, voxel_generator, box_coder)
        return net
    def predcit(self, pointclouds):
        t0=time.time()
        #滤处雷达附近的噪点
        mask=np.sqrt(np.square(pointclouds[:,:2]).sum(axis=1))>5
        pointclouds=pointclouds[mask]
        ret = self.voxel_generator.generate(pointclouds, max_voxels=12000)
        voxels = ret['voxels']
        coords = ret['coordinates']
        num_points = ret['num_points_per_voxel']
        # print('befor filter voxels shape is ',voxels.shape)
        if self.bg_filter is not None:
            voxels_mask = self.bg_filter.filter_bg(voxels,num_points, coords)
            voxels = voxels[voxels_mask]
            coords = coords[voxels_mask]
            num_points = num_points[voxels_mask]
            # print("after filter voxels shape is ", voxels.shape)
        voxels, num_points, coords, voxel_mask = voxel_padding(voxels, num_points,
                                                               coords, max_voxel_num=self.max_voxel_num)
        # print('after padding voxels shape is ', voxels.shape)

        # mask_points = np.logical_not(np.all(voxels.reshape(-1, 4) == 0, axis=1))
        # points = voxels.reshape(-1, 4)[mask_points]

        example = {
            "anchors": self.anchors,
            "voxels": voxels[np.newaxis, ...],
            "num_points": num_points[np.newaxis, ...],
            "coordinates": coords[np.newaxis, ...],
            'voxel_mask': voxel_mask[np.newaxis, ...],
            "metadata": [{"image_idx": '000000'}]}
        if self.anchors_area >= 0:
            # 计算每个grid map坐标位置是否有pillars（非空）
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coords, tuple(self.grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            # 计算每个anchor_bev占有的非空的pillars
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, self.anchors_bv, self.voxel_size, self.pc_range, self.grid_size)
            anchors_mask = anchors_area >= self.anchors_area
            if anchors_mask.sum() < 1:
                anchors_mask = np.zeros(anchors_area.shape[0], dtype=np.bool)
                print("anchors_mask is zero")
            example['anchors_mask'] = anchors_mask



        example_list = example_to_tensorlist(example,device=self.device, float_type=self.float_dtype)

        with torch.no_grad():
            boxes,scores = self.net(*example_list)[0][:2]
            boxes=boxes.detach().cpu().numpy()
            scores=scores.detach().cpu().numpy()
            print("current frame process time is {:.3f}ms".format((time.time()-t0)*1000))
        return boxes,scores



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_dir", type=str, default="./weights/pointpillars",help='输入模型目录')
    parse.add_argument("--bg_filter",action='store_true',default=False,help="是否集成背景滤波")
    parse.add_argument("--tensorrt",action='store_true',default=False,help="是否使用tensorrt引擎")
    parse.add_argument("--bg_dir",type=str, default='./data/1212',help="输入背景表目录")
    parse.add_argument("--anchors_area", type=int, default=10, help="输入anchor_area阈值")
    # parse.set_defaults(bg_filter=False)
    # parse.set_defaults(tensorrt=False)
    args=parse.parse_args()

    second_ros=SecondROS(model_dir=args.model_dir,bg_filter=args.bg_filter,
                         is_tensorrt=args.tensorrt,bg_dir=args.bg_dir,anchors_area=args.anchors_area)