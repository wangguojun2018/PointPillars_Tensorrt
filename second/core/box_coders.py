from second.core.box_coders_np import GroundBox3dCoder
from second.core import box_torch_ops


class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode,
                                               self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode,
                                               self.linear_dim)
