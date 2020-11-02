import torch
from libs.pub_models.bagnet import *


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def test_bagnet():
    pretrained = False

    models = [
        bagnet9,
        bagnet17,
        bagnet33,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bagnet9 or weight_count == 15688744)
        assert (model != bagnet17 or weight_count == 16213032)
        assert (model != bagnet33 or weight_count == 18310184)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))
