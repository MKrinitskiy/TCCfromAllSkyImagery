# pure-torch implementation of elastic transformations
# import numpy as np
import torch as t
# from .bilinear_interpolation import *
import torch.nn.functional as F
from .gather_nd import gather_nd


def rank(x):
    return len(x.shape)


class ElasticTransformer:


    def __init__(self, img_size:tuple, batch_size:int, flow_blur_size: int = 13, flow_initial_size: tuple = (8, 8),
                 flow_displacement_range: int = 8):
        '''
        Instantiate an ElasticTransformer
        :param img_size: Expected size (H,W,C) of images. Note: it is fixed, and flow fields will be generated according
        to this expected size of images. Images of different sizes will produce an exception at runtime.
        :param batch_size: expected batch size of images. Note: it is fixed, and flow field will be generated according
        to this batch size. Batches of images of different batch_size will produce an exception at runtime.
        :param flow_blur_size: the size of a squared mean-filtering window for the flow field
        :param flow_initial_size: size of the initial crude flow field (a subject to be resized to the size of
        transformed images)
        :param flow_displacement_range: absolute max value of the displacement for the flow field
        '''

        self.flow_initial_size = flow_initial_size
        self.flow_displacement_range = flow_displacement_range

        cuda = True if t.cuda.is_available() else False
        if cuda:
            self.device = t.device('cuda:0')
        else:
            self.device = t.device('cpu')

        self.batch_size = batch_size
        self.fixed_img_size = True
        self.img_size = img_size
        H = self.img_size[1]
        W = self.img_size[2]
        self.max_y = H - 1
        self.max_x = W - 1
        y = t.linspace(0.0, H - 1.0, H, dtype=t.float32, device=self.device)
        x = t.linspace(0.0, W - 1.0, W, dtype=t.float32, device=self.device)

        ygrid, xgrid = t.meshgrid([y, x])
        xgrid = xgrid.unsqueeze(0).unsqueeze(0)
        ygrid = ygrid.unsqueeze(0).unsqueeze(0)
        self.xgrid = t.repeat_interleave(xgrid, self.batch_size, 0)
        self.ygrid = t.repeat_interleave(ygrid, self.batch_size, 0)

        self.flow_blur_size = flow_blur_size
        flow_smoothing_kernel = t.ones(size=(flow_blur_size, flow_blur_size), dtype=t.float32, device=self.device)
        flow_smoothing_kernel = flow_smoothing_kernel / flow_smoothing_kernel.sum()
        flow_smoothing_kernel = flow_smoothing_kernel.unsqueeze(0)
        flow_smoothing_kernel = flow_smoothing_kernel.unsqueeze(0)
        flow_smoothing_kernel = t.repeat_interleave(flow_smoothing_kernel, 2, 0)
        self.flow_smoothing_kernel = flow_smoothing_kernel

        batch_idx = t.arange(0, self.batch_size, device=self.device)
        batch_idx = t.reshape(batch_idx, (-1, 1, 1, 1, 1))
        batch_idx = t.repeat_interleave(batch_idx, self.img_size[0], 1)
        batch_idx = t.repeat_interleave(batch_idx, self.img_size[1], 2)
        batch_idx = t.repeat_interleave(batch_idx, self.img_size[2], 3)
        self.batch_idx = batch_idx

        channels = self.img_size[0]
        channel_idx = t.arange(0, channels, device=self.device)
        channel_idx = t.reshape(channel_idx, (1, -1, 1, 1, 1))
        channel_idx = t.repeat_interleave(channel_idx, self.batch_size, 0)
        channel_idx = t.repeat_interleave(channel_idx, self.img_size[1], 2)
        channel_idx = t.repeat_interleave(channel_idx, self.img_size[2], 3)
        self.channel_idx = channel_idx


    def generate_flow(self):
        rand_x = t.rand([self.batch_size, 1] + list(self.flow_initial_size), dtype=t.float32, device=self.device)
        rand_x = (rand_x - 0.5) * 2.0 * self.flow_displacement_range
        rand_x[:, :, 0, :] = 0.0
        rand_x[:, :, -1, :] = 0.0
        rand_x[:, :, :, 0] = 0.0
        rand_x[:, :, :, -1] = 0.0
        rand_y = t.rand([self.batch_size, 1] + list(self.flow_initial_size), dtype=t.float32, device=self.device)
        rand_y = (rand_y - 0.5) * 2.0 * self.flow_displacement_range
        rand_y[:, :, 0, :] = 0.0
        rand_y[:, :, -1, :] = 0.0
        rand_y[:, :, :, 0] = 0.0
        rand_y[:, :, :, -1] = 0.0
        rand_x_scaled = F.interpolate(rand_x, size=(self.img_size[1], self.img_size[2]), mode='bilinear')
        rand_y_scaled = F.interpolate(rand_y, size=(self.img_size[1], self.img_size[2]), mode='bilinear')
        flow = t.cat([rand_x_scaled, rand_y_scaled], dim=1)
        flow_smoothed = F.conv2d(flow, self.flow_smoothing_kernel, bias=None, stride=1,
                                 padding=(self.flow_blur_size - 1) // 2,
                                 groups=2)

        return flow_smoothed


    def transform(self, img:t.Tensor, flow:t.Tensor):
        xt = flow[:, 0, :, :].unsqueeze(1)
        yt = flow[:, 1, :, :].unsqueeze(1)
        xt = self.xgrid + xt
        yt = self.ygrid + yt

        x0 = t.floor(xt).long()
        x1 = x0 + 1
        y0 = t.floor(yt).long()
        y1 = y0 + 1

        x0 = t.clamp(x0, 0, self.max_x)
        x1 = t.clamp(x1, 0, self.max_x)
        y0 = t.clamp(y0, 0, self.max_y)
        y1 = t.clamp(y1, 0, self.max_y)

        x0r = t.repeat_interleave(x0, 3, 1)
        y0r = t.repeat_interleave(y0, 3, 1)
        x1r = t.repeat_interleave(x1, 3, 1)
        y1r = t.repeat_interleave(y1, 3, 1)

        x0r = x0r.unsqueeze(-1)
        y0r = y0r.unsqueeze(-1)
        x1r = x1r.unsqueeze(-1)
        y1r = y1r.unsqueeze(-1)

        idxA = t.cat([self.batch_idx, self.channel_idx, y0r, x0r], dim=-1)
        gatheredA = gather_nd(img, idxA)

        idxB = t.cat([self.batch_idx, self.channel_idx, y1r, x0r], dim=-1)
        gatheredB = gather_nd(img, idxB)

        idxC = t.cat([self.batch_idx, self.channel_idx, y0r, x1r], dim=-1)
        gatheredC = gather_nd(img, idxC)

        idxD = t.cat([self.batch_idx, self.channel_idx, y1r, x1r], dim=-1)
        gatheredD = gather_nd(img, idxD)

        wa = (x1 - xt) * (y1 - yt)
        wb = (x1 - xt) * (yt - y0)
        wc = (xt - x0) * (y1 - yt)
        wd = (xt - x0) * (yt - y0)
        wa = t.repeat_interleave(wa, 3, 1)
        wb = t.repeat_interleave(wb, 3, 1)
        wc = t.repeat_interleave(wc, 3, 1)
        wd = t.repeat_interleave(wd, 3, 1)
        A = wa * gatheredA
        B = wb * gatheredB
        C = wc * gatheredC
        D = wd * gatheredD

        img_transformed = A + B + C + D

        return img_transformed

