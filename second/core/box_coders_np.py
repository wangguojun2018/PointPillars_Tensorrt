from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from second.core import box_np_ops

class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, vec_encode=False, custom_ndim=0):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode
        self.custom_ndim = custom_ndim

    @property
    def code_size(self):
        res = 8 if self.vec_encode else 7
        return self.custom_ndim + res

    def _encode(self, boxes, anchors):
        return box_np_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        return box_np_ops.second_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)



