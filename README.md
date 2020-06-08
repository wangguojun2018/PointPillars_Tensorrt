# Lidar-detection
ONLY support python 3.6+, pytorch 1.0.0+. Tested in Ubuntu 16.04.
## 雷达驱动
通过更改雷达驱动使驱动使之可以输出通道ring信息，为算法提供支持
使用方法参考[rslidar](https://index.ros.org/r/robosense/#kinetic)
## Install

### 1. Clone code

```bash
git clone https://https://github.com/tt-leader/Lidar-detection.git
```

### 2. Install dependence python packages

```bash
pip3 install numba pyntcloud pyyaml rospkg pyquaternion protobuf
```
`安装ros_numpy`:   
```angular2
git clone https://github.com/eric-wieser/ros_numpy
cd ros_numpy && python setup.py install
```
`LLVM_CONFIG`找不到的问题：
```angular2
sudo apt-get install llvm-8
export LLVM_CONFIG=/usr/bin/llvm-config-8
pip3 install numba
```
`No lapack/blas resources found` 问题：   
```bash
apt-get install gfortran libopenblas-dev liblapack-dev
pip3 install scipy
```

Follow instructions in [spconv](https://github.com/wangguojun2018/spconv) to install spconv. 

`can not find CUDNN`问题：添加软连接
```angular2
sudo ln -s libcudnn.so.7 libcudnn.so
```
Follow instructions in [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to install torch2trt
### 3. 安装ros 依赖：
```bash
apt install ros-melodic-rospy ros-melodic-ros-base ros-melodic-sensor-msgs ros-melodic-jsk-recognition-msgs ros-melodic-visualization-msgs
```
### 4. Setup cuda for numba

you need to add following environment variable for numba.cuda, you can add them to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/aarch64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

### 5. add Lidar-detection/ to PYTHONPATH

## Usage

### 生成tensorrt引擎  
1. 进入工程根目录
```
cd ./Lidar-detection/second
```
2. 生成tensorrt引擎  
```
python utils/convert2rt.py --model_dir ./weights/pointpillarsBG --bg_filter  
```

3. 启动ros推理节点  
```
python inference_ros.py --tensorrt  --model_dir ./weights/pointpillarsBG --bg_filter --tensorrt
```
4. 启动rviz可视化  
```
rviz -d rviz/lidar_detection.rviz
```
## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
