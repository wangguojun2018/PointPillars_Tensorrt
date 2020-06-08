# 模型权重目录说明
1. 工程提供两种模型，分别是基于原始数据训练和背景滤波后数据训练（带*reduced*后缀为背景滤波的模型）
2. 每个模型目录必须包含以下四个文件  
`pipeline.config` 为模型参数配置的文件  
`voxelnet.tckpt` 为模型权重文件  
`pfe.trt` 为voxel特征提取子网络引擎  
`rpn.trt` 为rpn子网络引擎
