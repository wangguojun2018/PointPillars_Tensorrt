import os
import math
import numpy as np
import glob
from pyntcloud import PyntCloud
from pathlib import Path
import argparse
from second.utils.preprocess import get_paddings_indicator_np
from second.protos import pipeline_pb2
from google.protobuf import text_format
from second.builder import voxel_builder
ROT_TRANS=np.array([[-0.984808,0.173648,0],
[-0.173648, -0.984808, 0],[0 ,0,1]])

class bg_filter(object):
    def __init__(self,voxel_cfg,num_point=3,diff=0.3,num_path=None,var_path=None,is_statistic=False):
        self._config=voxel_cfg
        self._statistic=is_statistic
        self._voxel_size=self._config.voxel_size[0]
        self._num=num_point
        self._diff=diff
        self._num_path=num_path
        self._val_path=var_path
        self._xmin = self._config.point_cloud_range[0]
        self._ymin = self._config.point_cloud_range[1]
        self._xmax=self._config.point_cloud_range[3]
        self._ymax = self._config.point_cloud_range[4]
        self._row = math.ceil((self._xmax-self._xmin)/ self._voxel_size)
        self._column = math.ceil((self._ymax-self._ymin)/ self._voxel_size)

        # print("row is ",self._row,"column is ",self._column)
        self._table_num = np.zeros((self._row, self._column, 2), dtype=np.float)
        if os.path.exists(self._num_path):
            self._table_num[:,:,0]=np.loadtxt(self._num_path)
        if os.path.exists(self._val_path):
            self._table_num[:,:,1]=np.loadtxt(self._val_path)
    def generate_table(self,bg_pcd_dir,voxel_generator):
        bg_pcd_files=glob.glob(bg_pcd_dir+'/*.pcd')
        # if self._statistic:
        #     assert len(bg_pcd_files)>1,"if statistic,bg_pcd files must greater than 1"
        # else:
        #     assert len(bg_pcd_files)==1,"if not statistic,bg_pcd files must equal 1"
        table_num_list=[]
        for pcd_file in bg_pcd_files:
            table_num = np.zeros(self._table_num.shape[:2])
            cloud=PyntCloud.from_file(pcd_file)
            points=cloud.points.values
            ret = voxel_generator.generate(points, max_voxels=12000)
            coords = ret['coordinates']
            num_points = ret['num_points_per_voxel']
            x_cor = coords[:,2]
            y_cor = coords[:,1]
            table_num[x_cor, y_cor] = num_points
            table_num_list.append(table_num)
        table_num_list=np.stack(table_num_list,axis=0)
        self._table_num[:,:,0]=np.mean(table_num_list,axis=0)
        self._table_num[:,:,1]=np.std(table_num_list,axis=0)
        np.savetxt(self._num_path,self._table_num[:,:,0])
        np.savetxt(self._val_path, self._table_num[:, :, 1])
        print("generate statistic table successful!")
    def filter_bg(self,voxels,num_points,coords):
        coords_x=coords[:,2]
        coords_y=coords[:,1]
        z_diff = voxels[:, :, 2] - voxels[:, :, 2].min(axis=1)[:, np.newaxis]
        mask = get_paddings_indicator_np(num_points, voxels.shape[1], axis=0)  # (K,T)
        z_diff = (z_diff * mask).max(axis=1)
        if self._statistic:
            voxels_mask = np.abs(num_points - self._table_num[coords_x, coords_y, 0]) >= self._num * \
                          self._table_num[coords_x, coords_y, 1]

        else:
            voxels_mask = np.abs(num_points -
                                 self._table_num[coords_x, coords_y, 0]) >= self._num

        diff_mask = np.logical_and(num_points >= 2, z_diff >= self._diff)
        # diff_mask = np.logical_or(np.logical_and(num_points == 1, self._table_num[coords_x, coords_y, 0] == 0),
        #                           np.logical_and(num_points >= 2, z_diff >= self._diff))
        voxels_mask = np.logical_and(voxels_mask, diff_mask)
        return voxels_mask


if __name__=="__main__":
    import os
    import getpass
    usr_name = getpass.getuser()
    os.chdir('/home/'+usr_name+'/source_code/py/second.pytorch/second')
    config_path='./configs/xyres_28_huituo.config'
    parse=argparse.ArgumentParser()
    parse.add_argument("--num",type=float,default=3)
    parse.add_argument("--diff", type=float, default=0.25)
    args=parse.parse_args()

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    voxel_cfg=config.model.second.voxel_generator
    voxel_cfg.point_cloud_range[:]=[-29.84, -66.32, -9, 59.76, 59.12, 3]
    voxel_generator=voxel_builder.build(voxel_cfg)

    data_dir="/home/"+usr_name+"/dataset/roadside/1212/rslidar"
    data_reduced_dir = "/home/"+ usr_name+"/dataset/roadside/1113/training/velodyne_reduced"
    bg_file="./data/1212"
    bin_files=glob.glob(data_dir+"/*.pcd")
    bin_files.sort()
    myfilter = bg_filter(voxel_generator.config, num_path='./data/1212/num_table.txt',
                         var_path='./data/1212/var_table.txt',num_point=args.num,
                         is_statistic=False,diff=args.diff)
    # myfilter.generate_table(bg_file,voxel_generator)
    # myfilter=bg_filter_ring(voxel_cfg,z_path="./data/1113/z_table.txt",azimuth_res=0.2,
    #                         laser_angle_path="./data/1113/laser_angles.txt",diff_z=args.num)
    # myfilter.generate_table(bg_file+"/000082.pcd")
    # vis = o3d.visualization.Visualizer()
    # vis=o3d.Visualizer()
    # vis.create_window()
    # vis.get_render_option().load_from_json('visualizer.json')
    # vis.register_key_callback(ord("a"),key_callback)
    update=False
    max_voxel_num=0
    # while bin_number<len(bin_files):
    filter_times=[]
    for bin_file in bin_files:
        # bin_file=bin_files[bin_number]

        # reduced_bin = data_dir + '/' + Path(bin_file).stem + '.bin'
        print("current bin is ",Path(bin_file).stem)
        # points=np.fromfile(bin_file,dtype=np.float32).reshape(-1,4)
        # pcd = o3d.io.read_point_cloud(bin_file, format='pcd')
        # points = np.asarray(pcd.points)
        cloud=PyntCloud.from_file(bin_file)
        points=cloud.points.values
        # points=points[points[:,4]==31]
        # print("points shape is ",points.shape)
        ret = voxel_generator.generate(points, max_voxels=12000)
        voxels = ret['voxels']
        coords = ret['coordinates']
        num_points = ret['num_points_per_voxel']
        print('befor filter voxels shape is ', voxels.shape)
        print("after filter points number is ", points.shape[0])
        voxels_mask = myfilter.filter_bg(voxels,num_points,coords)
        voxels = voxels[voxels_mask]
        coords = coords[voxels_mask]
        num_points = num_points[voxels_mask]
        print("after filter voxels shape is ", voxels.shape)
        # # #
        points=voxels.reshape(-1,voxels.shape[-1])[:,:4]
        mask_points=np.logical_not(np.all(points==0,axis=1))
        points=points[mask_points]
        print("after filter points number is ", points.shape[0])








