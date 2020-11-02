from libs.elastic_transform import *
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .FCN_resnet_blocks import BasicBlock
import numpy as np
from torch.nn import init


class SIAmodel_FCN(nn.Module):


    def __init__(self, args, classes_num=9, input_data_length=606, verbose=False):
        super(SIAmodel_FCN, self).__init__()

        self.dump_activations = False

        initial_slope = 0.05

        self.model_type = args.model_type
        self.name = 'encoder'
        self.verbose = verbose
        self.debug = args.debug

        self.modules_list = nn.ModuleList()

        self.watch_activations_modules_names = []
        self.watch_activations_modules_modules2names = {}
        watch_activations_module_idx = 1
        watch_activations_module_name_pattern = 'module_%03d'

        in_features = input_data_length
        out_features = int(np.power(2, np.ceil(np.log(input_data_length)/np.log(2.0))))
        while out_features > classes_num:
            drop = nn.Dropout(p=0.5)
            linear = nn.Linear(in_features, out_features)
            init.kaiming_normal_(linear.weight, a=initial_slope)
            bn = nn.BatchNorm1d(out_features)
            act = nn.PReLU(init=initial_slope)
            self.modules_list.extend([drop, linear, bn, act])
            mname = watch_activations_module_name_pattern % watch_activations_module_idx
            watch_activations_module_idx += 1
            self.watch_activations_modules_names.append(mname)
            self.watch_activations_modules_modules2names[self.modules_list[-1]] = mname
            in_features = out_features
            out_features = out_features // 2

        # region fully-connected subnet after 1D resnet blocks
        if self.model_type in ['PC', 'OR']:
            l3 = nn.Linear(16, classes_num)
            init.kaiming_normal_(l3.weight, a=initial_slope)
        elif self.model_type == 'ORbin':
            l3 = nn.Linear(16, 1)
            init.kaiming_normal_(l3.weight, a=initial_slope)
        self.modules_list.extend([l3])
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