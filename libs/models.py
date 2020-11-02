from libs.elastic_transform import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_blocks import BasicBlock
import numpy as np
from torch.nn import init
from .pub_models.pyramidnet import *
from functools import reduce


class SIAmodel(nn.Module):
    def __init__(self, args, classes_num=9, verbose=False):
        super(SIAmodel, self).__init__()

        activation = nn.LeakyReLU(inplace=True)

        self.model_type = args.model_type
        self.name = 'encoder'
        self.verbose = verbose
        self.debug = args.debug
        self.blocks_num = args.blocks_num
        img_size = (3,args.img_size,args.img_size)
        curr_spatial_size = np.array([args.img_size,args.img_size])

        x_channels = 3
        self.modules_list = nn.ModuleList()
        # self.modules_names = []

        self.watch_activations_modules_names = []
        self.watch_activations_modules_modules2names = {}
        watch_activations_module_idx = 1
        watch_activations_module_name_pattern = 'module_%03d'

        out_x_channels = 8  # starting value, not used

        for block_number in range(args.blocks_num):
            # region Conv block: spatial x -> x/2; channels x -> x*2
            out_x_channels = out_x_channels * 2  # 16
            rnb1 = BasicBlock(x_channels, x_channels, activation=activation)
            rnb2 = BasicBlock(x_channels, x_channels, activation=activation)
            downsample_conv = nn.Conv2d(x_channels, out_x_channels, kernel_size=3, padding=1, stride=2, bias=True)
            init.kaiming_normal_(downsample_conv.weight, activation.negative_slope)
            downsample_bn = nn.BatchNorm2d(out_x_channels)
            downsample = nn.Sequential(downsample_conv, downsample_bn)
            rnb3 = BasicBlock(x_channels, out_x_channels, downsample=downsample, stride=2, activation=activation)
            x_channels = out_x_channels  # 16
            # self.modules_names.extend(['resnet_block_1', 'resnet_block_2', 'resnet_block_3'])
            self.modules_list.extend([rnb1, rnb2, rnb3])
            curr_spatial_size = (curr_spatial_size/2).astype(np.int32)
            # region for activations debug
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname
            # endregion
            # endregion

        if np.product(curr_spatial_size) >= x_channels:
            # do reshape to a vector and channels x_channels -> 1
            # region last Conv block: spatial 16 -> 16; channels 256 -> 1
            out_x_channels = 1
            rnb1 = BasicBlock(x_channels, x_channels, activation=activation)
            rnb2 = BasicBlock(x_channels, x_channels, activation=activation)
            conv1x1_3 = nn.Conv2d(x_channels, out_x_channels, kernel_size=1, stride=1, bias=False)
            init.kaiming_normal_(conv1x1_3.weight, activation.negative_slope)
            bn_3 = nn.BatchNorm2d(out_x_channels)
            act_3 = activation
            self.modules_list.extend([rnb1, rnb2, conv1x1_3, bn_3, act_3])
            # region for activations debug
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname
            # endregion
            # endregion
            self.transfer_type = '1x1conv'
        else:
            # do global average pooling and reshape to a vector
            # region last Conv block: spatial 16 -> 1; channels 256 -> 256
            rnb1 = BasicBlock(x_channels, x_channels, activation=activation)
            rnb2 = BasicBlock(x_channels, x_channels, activation=activation)
            self.modules_list.extend([rnb1, rnb2])
            # region for activations debug
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname
            # endregion
            # endregion
            self.transfer_type = 'gap'




        self.modules_list_fc = nn.ModuleList()

        # region fully-connected subnet
        reshaped_vec_length = np.product(curr_spatial_size)
        if self.transfer_type == '1x1conv':
            reshaped_vec_length = np.product(curr_spatial_size)
        elif self.transfer_type == 'gap':
            reshaped_vec_length = x_channels
        l1 = nn.Linear(reshaped_vec_length, 64)
        init.kaiming_normal_(l1.weight, activation.negative_slope)
        bn1 = nn.BatchNorm1d(64)
        act1 = activation
        l2 = nn.Linear(64, 16)
        init.kaiming_normal_(l2.weight, activation.negative_slope)
        bn2 = nn.BatchNorm1d(16)
        act2 = activation
        l3 = nn.Linear(16, classes_num)
        init.kaiming_normal_(l3.weight, activation.negative_slope)
        self.modules_list_fc.extend([l1, bn1, act1, l2, bn2, act2, l3])
        # region for activations debug
        for m in [act1, act2]:
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[m] = mname
        # endregion
        # endregion

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img, msk, dump_activations = False):
        x = t.mul(img, msk)

        if self.debug & dump_activations:
            activations = {}
            activations['module_0_input'] = x.detach().cpu().numpy()

        # for m,mname in zip(self.modules_list, self.modules_names):
        for m in self.modules_list:
            x = m(x)
            if (self.debug & dump_activations & (m in self.watch_activations_modules_modules2names.keys())):
                mname = self.watch_activations_modules_modules2names[m]
                activations[mname] = x.detach().cpu().numpy()

        if self.transfer_type == '1x1conv':
            x = x.view((x.shape[0], -1))
        elif self.transfer_type == 'gap':
            x = x.mean(dim=[2,3], keepdim=False)
            x = x.view((x.shape[0], -1))
        for idx,m in enumerate(self.modules_list_fc):
            x = m(x)
            if (self.debug & dump_activations & (m in self.watch_activations_modules_modules2names.keys())):
                mname = self.watch_activations_modules_modules2names[m]
                activations[mname] = x.detach().cpu().numpy()

        x = x.view(x.shape[0], -1)
        if self.model_type == 'PC':
            x = F.softmax(x, dim=-1)
        elif self.model_type == 'OR':
            x = t.sigmoid(x)

        if self.debug & dump_activations:
            return x, activations
        else:
            return x



class SIAmodel_PyramidNet(nn.Module):
    def __init__(self, args, classes_num=9, verbose=False):
        super(SIAmodel_PyramidNet, self).__init__()

        # self.activation = nn.PReLU(init=0.02)
        self.in_size = (args.img_size, args.img_size)
        self.classes_num = classes_num

        self.model_type = args.model_type
        self.name = 'encoder'
        self.verbose = verbose
        self.debug = args.debug
        self.blocks_num = args.blocks_num

        x_channels = 3
        self.modules_list = nn.ModuleList()

        self.watch_activations_modules_names = []
        self.watch_activations_modules_modules2names = {}
        watch_activations_module_idx = 1
        watch_activations_module_name_pattern = 'module_%03d'

        alpha = args.pnet_alpha
        blocks = args.pnet_blocks
        if blocks == 10:
            layers = [1, 1, 1, 1]
        elif blocks == 12:
            layers = [2, 1, 1, 1]
        elif blocks == 14:
            layers = [2, 2, 1, 1]
        elif blocks == 16:
            layers = [2, 2, 2, 1]
        elif blocks == 18:
            layers = [2, 2, 2, 2]
        elif blocks == 34:
            layers = [3, 4, 6, 3]
        elif blocks == 50:
            layers = [3, 4, 6, 3]
        elif blocks == 101:
            layers = [3, 4, 23, 3]
        elif blocks == 152:
            layers = [3, 8, 36, 3]
        elif blocks == 200:
            layers = [3, 24, 36, 3]

        init_block_channels = 64

        self.modules_list = nn.ModuleList()

        growth_add = float(alpha) / float(sum(layers))
        channels = reduce(
                lambda xi, yi: xi + [[(i + 1) * growth_add + xi[-1][-1] for i in list(range(yi))]],
                layers,
                [[init_block_channels]])[1:]
        channels = [[int(round(cij)) for cij in ci] for ci in channels]
        if blocks < 50:
            bottleneck = False
        else:
            bottleneck = True
            channels = [[cij * 4 for cij in ci] for ci in channels]

        self.features = nn.Sequential()
        self.features.add_module("init_block", PyrInitBlock(in_channels=x_channels, out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                unit = PyrUnit(in_channels=in_channels, out_channels=out_channels,
                               stride=stride, bottleneck=bottleneck)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(in_channels=in_channels))
        self.features.add_module("AvgPool2d", nn.AvgPool2d(kernel_size=7, stride=1))
        self.features.add_module("AvgPool2d_activ", PreResActivation(in_channels=in_channels))

        if self.model_type in ['PC', 'OR']:
            self.output = nn.Linear(in_features=in_channels, out_features=self.classes_num)
        elif self.model_type == 'ORbin':
            self.output = nn.Linear(in_features=in_channels, out_features=1)
        # self.modules_list.append(self.output)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, img, msk):
        x = t.mul(img, msk)

        x = self.features(x)

        x = torch.mean(x, dim=[2,3])

        x = x.view(x.size(0), -1)

        x = self.output(x)

        if self.model_type == 'PC':
            x = F.softmax(x, dim=-1)
        elif self.model_type == 'OR':
            x = t.sigmoid(x)

        return x