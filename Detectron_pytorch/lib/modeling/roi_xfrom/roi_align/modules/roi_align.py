from torch.nn.functional import avg_pool2d, max_pool2d
from torch.nn.modules.module import Module

from ..functions.roi_align import RoIAlignFunction


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        return RoIAlignFunction(
            self.aligned_height,
            self.aligned_width,
            self.spatial_scale,
            self.sampling_ratio,
        )(features, rois)


class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(
            self.aligned_height + 1,
            self.aligned_width + 1,
            self.spatial_scale,
            self.sampling_ratio,
        )(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x = RoIAlignFunction(
            self.aligned_height + 1,
            self.aligned_width + 1,
            self.spatial_scale,
            self.sampling_ratio,
        )(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
