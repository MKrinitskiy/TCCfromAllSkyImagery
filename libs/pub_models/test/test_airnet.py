import torch
from libs.pub_models.airnet import *





def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def test_airnet():
    pretrained = False

    models = [
            airnet50_1x64d_r2,
            airnet50_1x64d_r16,
            airnet101_1x64d_r2,
            ]

    for model in models:
        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != airnet50_1x64d_r2 or weight_count == 27425864)
        assert (model != airnet50_1x64d_r16 or weight_count == 25714952)
        assert (model != airnet101_1x64d_r2 or weight_count == 51727432)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))