import torch
from libs.pub_models.alphapose_coco import *



def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count



def test_alphapose_fastseresnet101b_coco():
    in_size = (256, 192)
    keypoints = 17
    return_heatmap = False
    pretrained = False

    models = [
            alphapose_fastseresnet101b_coco,
            ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, return_heatmap=return_heatmap)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != alphapose_fastseresnet101b_coco or weight_count == 59569873)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert ((y.shape[0] == batch) and (y.shape[1] == keypoints))
        if return_heatmap:
            assert ((y.shape[2] == x.shape[2] // 4) and (y.shape[3] == x.shape[3] // 4))
        else:
            assert (y.shape[2] == 3)