from libs.elastic_transform import *
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .FCN_resnet_blocks import BasicBlock
import numpy as np
from torch.nn import init


class SIAmodel_FCN_residual(nn.Module):
    def __init__(self, args, classes_num=9, input_data_length=606, verbose=False):
        super(SIAmodel_FCN_residual, self).__init__()

        self.dump_activations = False

        initial_slope = 0.05
        # activation = nn.PReLU(init=initial_slope)
        # activation = nn.LeakyReLU()

        self.model_type = args.model_type
        self.name = 'encoder'
        self.verbose = verbose
        self.debug = args.debug
        self.blocks_num = args.blocks_num
        # curr_data_length = input_data_length

        self.modules_list = nn.ModuleList()
        # self.modules_names = []

        self.watch_activations_modules_names = []
        self.watch_activations_modules_modules2names = {}
        watch_activations_module_idx = 1
        watch_activations_module_name_pattern = 'module_%03d'

        in_features = input_data_length
        out_features = int(np.power(2, np.ceil(np.log(input_data_length)/np.log(2.0))))

        drop1 = nn.Dropout(p=0.5)
        drop2 = nn.Dropout(p=0.5)
        drop3 = nn.Dropout(p=0.5)
        rnb1 = BasicBlock(in_features, in_features, initial_slope=initial_slope)
        rnb2 = BasicBlock(in_features, in_features, initial_slope=initial_slope)
        downsample_linear = nn.Linear(in_features, out_features)
        init.kaiming_normal_(downsample_linear.weight, a=initial_slope)
        downsample_bn = nn.BatchNorm1d(out_features)
        downsample = nn.Sequential(downsample_linear, downsample_bn)
        rnb3 = BasicBlock(in_features, out_features, downsample=downsample, initial_slope=initial_slope)
        self.modules_list.extend([drop1, rnb1, drop2, rnb2, drop3, rnb3])
        mname = watch_activations_module_name_pattern % watch_activations_module_idx
        watch_activations_module_idx += 1
        self.watch_activations_modules_names.append(mname)
        self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname

        for block_number in range(args.blocks_num-1):
            # region Conv block: spatial x -> x/2; channels x -> x*2
            in_features = out_features
            out_features = out_features // 2
            drop1 = nn.Dropout(p=0.5)
            drop2 = nn.Dropout(p=0.5)
            drop3 = nn.Dropout(p=0.5)
            rnb1 = BasicBlock(in_features, in_features, initial_slope=initial_slope)
            rnb2 = BasicBlock(in_features, in_features, initial_slope=initial_slope)
            downsample_linear = nn.Linear(in_features, out_features)
            init.kaiming_normal_(downsample_linear.weight, a = initial_slope)
            downsample_bn = nn.BatchNorm1d(out_features)
            downsample = nn.Sequential(downsample_linear, downsample_bn)
            rnb3 = BasicBlock(in_features, out_features, downsample=downsample, initial_slope=initial_slope)
            self.modules_list.extend([drop1, rnb1, drop2, rnb2, drop3, rnb3])
            # region for activations debug
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname
            # endregion
            # endregion

        # region fully-connected subnet after 1D resnet blocks
        in_features = out_features
        out_features = out_features // 2
        drop1 = nn.Dropout(p=0.2)
        drop2 = nn.Dropout(p=0.2)
        l1 = nn.Linear(in_features, 64)
        init.kaiming_normal_(l1.weight, a=initial_slope)
        bn1 = nn.BatchNorm1d(64)
        act1 = nn.PReLU(init=initial_slope)
        l2 = nn.Linear(64, 16)
        init.kaiming_normal_(l2.weight, a=initial_slope)
        bn2 = nn.BatchNorm1d(16)
        act2 = nn.PReLU(init=initial_slope)
        if self.model_type in ['PC', 'OR']:
            l3 = nn.Linear(16, classes_num)
            init.kaiming_normal_(l3.weight, a=initial_slope)
        elif self.model_type == 'ORbin':
            l3 = nn.Linear(16, 1)
            init.kaiming_normal_(l3.weight, a=initial_slope)
        self.modules_list.extend([drop1, l1, bn1, act1, drop2, l2, bn2, act2, l3])
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


    def switch_dump_activations(self, switch_on = True):
        self.dump_activations = switch_on


    def forward(self, x):

        activations = {}
        if self.debug & self.dump_activations:
            activations['module_0_input'] = x.detach().cpu().numpy()

        # for m,mname in zip(self.modules_list, self.modules_names):
        for m in self.modules_list:
            x = m(x)
            if self.debug & self.dump_activations & (m in self.watch_activations_modules_modules2names.keys()):
                mname = self.watch_activations_modules_modules2names[m]
                activations[mname] = x.detach().cpu().numpy()

        if self.model_type == 'PC':
            x = F.softmax(x, dim=-1)
        elif self.model_type in ['OR', 'ORbin']:
            x = t.sigmoid(x)

        if self.debug & self.dump_activations:
            return x, activations
        else:
            return x