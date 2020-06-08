import numpy as np

class AnchorGenerator:
    @property
    def class_name(self):
        raise NotImplementedError

    @property
    def num_anchors_per_localization(self):
        raise NotImplementedError

    def generate(self, feature_map_size):
        raise NotImplementedError

    @property 
    def ndim(self):
        raise NotImplementedError


class AnchorGeneratorStride(AnchorGenerator):
    def __init__(self,
                 sizes=[1.6, 3.9, 1.56],
                 anchor_strides=[0.4, 0.4, 1.0],
                 anchor_offsets=[0.2, -39.8, -1.78],
                 rotations=[0, np.pi / 2],
                 class_name=None,
                 match_threshold=-1,
                 unmatch_threshold=-1,
                 custom_values=(),
                 dtype=np.float32):
        super().__init__()
        self._sizes = sizes
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def class_name(self):
        return self._class_name

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    def create_anchors_3d_stride(self,feature_size,
                                 sizes=[1.6, 3.9, 1.56],
                                 anchor_strides=[0.4, 0.4, 0.0],
                                 anchor_offsets=[0.2, -39.8, -1.78],
                                 rotations=[0, np.pi / 2],
                                 dtype=np.float32):
        """
        Args:
            feature_size: list [D, H, W](zyx)
            sizes: [N, 3] list of list or array, size of anchors, xyz

        Returns:
            anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
        """
        # almost 2x faster than v1
        x_stride, y_stride, z_stride = anchor_strides
        x_offset, y_offset, z_offset = anchor_offsets
        z_centers = np.arange(feature_size[0], dtype=dtype)
        y_centers = np.arange(feature_size[1], dtype=dtype)
        x_centers = np.arange(feature_size[2], dtype=dtype)
        z_centers = z_centers * z_stride + z_offset
        y_centers = y_centers * y_stride + y_offset
        x_centers = x_centers * x_stride + x_offset
        sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
        rotations = np.array(rotations, dtype=dtype)
        rets = np.meshgrid(
            x_centers, y_centers, z_centers, rotations, indexing='ij')
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
            rets[i] = rets[i][..., np.newaxis]  # for concat
        sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = np.tile(sizes, tile_size_shape)
        rets.insert(3, sizes)
        ret = np.concatenate(rets, axis=-1)
        return np.transpose(ret, [2, 1, 0, 3, 4, 5])
    def generate(self, feature_map_size):
        res = self.create_anchors_3d_stride(
            feature_map_size, self._sizes, self._anchor_strides,
            self._anchor_offsets, self._rotations, self._dtype)
        if len(self._custom_values) > 0:
            custom_ndim = len(self._custom_values)
            custom = np.zeros([*res.shape[:-1], custom_ndim])
            custom[:] = self._custom_values
            res = np.concatenate([res, custom], axis=-1)
        return res

    def generate_anchors(self,feature_map_size):
        ndim = len(feature_map_size)
        anchors = self.generate(feature_map_size)
        anchors = anchors.reshape([*feature_map_size, -1, self.ndim])
        anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
        anchors = anchors.reshape(-1, self.ndim)
        return anchors
    @property 
    def ndim(self):
        return 7 + len(self._custom_values)

    @property 
    def custom_ndim(self):
        return len(self._custom_values)
