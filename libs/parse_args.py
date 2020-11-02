import argparse, warnings, datetime, os
import numpy as np
from .service_defs import DoesPathExistAndIsFile


def parse_args(args):
    """ Parse the arguments.
        """
    parser = argparse.ArgumentParser(description='Simple training script for training an autoencoder for SPV clustering')

    # parser.add_argument('--srcdata-list', dest='srcdata_list', help='list of filenames (absolute paths, pickled)')
    parser.add_argument('--train-list', dest='train_list', help='list of filenames (absolute paths, pickled)', required=True)
    parser.add_argument('--test-list', dest='test_list', help='list of filenames (absolute paths, pickled)', required=True)
    parser.add_argument('--run-name', help='name for the current run (directories will be created based on this name)',
                        default='devel')
    parser.add_argument('--batch-size', help='Size of the batches.', dest='batch_size', type=int)
    parser.add_argument('--val-batch-size', help='Size of the batches for evaluation.', type=int)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int)
    parser.add_argument('--steps-per-epoch', help='Number of steps per epoch.', type=int)
    parser.add_argument('--val-steps', help='Number of steps per validation run.', type=int)
    # parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--debug', help='launch in DEBUG mode', dest='debug', action='store_true')
    parser.add_argument('--tta', help='apply test-time augmentations', dest='tta', action='store_true')
    parser.add_argument('--model-only', help='compose model only and output its description',
                        dest='model_only', action='store_true')
    parser.add_argument('--model-type',
                        help='model type for the TCC classification: either PC for pure classification or OR for ordinal regression',
                        dest='model_type', choices=['PC', 'OR'])
    parser.add_argument('--img-size', help='size of the images to resize them to', dest='img_size', type=int)
    parser.add_argument('--serial-datagen', help='switches off the parallel data generation', dest='serial_datagen',
                        action='store_true')
    parser.add_argument('--lr', help='starting learning rate', default=1.0e-4, type=float)

    parser.add_argument('--blocks-num', help='number of convolutional blocks (three resnet blocks each) of the model',
                        dest='blocks_num', type=int)
    parser.add_argument('--memcache', help='memcache images for faster loading', action='store_true')

    parser.add_argument('--pnet', help='directs to construct the network using PyramidNet architecture',
                        action='store_true')
    pyramidnetgroup = parser.add_argument_group()
    pyramidnetgroup.add_argument('--pnet-alpha', help='alpha parameter for the PyramidNet to be constructed',
                                 dest='pnet_alpha', type=float, default=450)
    pyramidnetgroup.add_argument('--pnet-blocks', help='number of blocks for the PyramidNet to be constructed',
                                 dest='pnet_blocks', type=int, default=10,
                                 choices=[10, 12, 14, 16, 18, 34, 50, 101, 152, 200])


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Resume training given a run name to resume from.', default=argparse.SUPPRESS)
    group.add_argument('--snapshot', help='Resume training given a run name to resume from.', default=argparse.SUPPRESS)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    assert DoesPathExistAndIsFile(parsed_args.train_list), 'train data index file not found:\n%s'%parsed_args.train_list
    assert DoesPathExistAndIsFile(parsed_args.test_list), 'test data index file not found:\n%s' % parsed_args.test_list

    if ('resume' not in parsed_args) & (not parsed_args.pnet):
        assert (parsed_args.img_size % np.power(2, parsed_args.blocks_num) == 0), 'in the current implementation,' \
                                                                                  'img_size must be completely divisible ' \
                                                                                  'by (2^blocks_num)'
        assert (parsed_args.img_size // np.power(2, parsed_args.blocks_num) >= 4), 'in the current implementation, ' \
                                                                                   'img_size must be greater or equal ' \
                                                                                   '4*(2^blocks_num)'

    return parsed_args