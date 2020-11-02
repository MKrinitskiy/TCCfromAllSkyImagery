import torch
from libs.pub_models.airnext import *


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count



def test_airnext():
    pretrained = False

    models = [
            airnext50_32x4d_r2,
            airnext101_32x4d_r2,
            airnext101_32x4d_r16,
            ]

    for model in models:
        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != airnext50_32x4d_r2 or weight_count == 27604296)
        assert (model != airnext101_32x4d_r2 or weight_count == 54099272)
        assert (model != airnext101_32x4d_r16 or weight_count == 45456456)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))
