import argparse, warnings, datetime, os
import numpy as np


def parse_args(args):
    """ Parse the arguments.
        """
    parser = argparse.ArgumentParser(description='Simple training script for training an autoencoder for SPV clustering')

    parser.add_argument('--srcdata-file', dest='srcdata_file', help='a file containing pickled pre-computed stats of '
                                                                    'images', default=argparse.SUPPRESS)
    parser.add_argument('--run-name', help='name for the current run (directories will be created based on this name)',
                        default='FCN_devel')
    parser.add_argument('--batch-size', help='Size of the batches.', dest='batch_size', default=16, type=int)
    parser.add_argument('--val-batch-size', help='Size of the batches for evaluation.', default=32, type=int)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)
    parser.add_argument('--steps-per-epoch', help='Number of steps per epoch.', type=int)
    parser.add_argument('--val-steps', help='Number of steps per validation run.', type=int, default=100)
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--debug', help='launch in DEBUG mode', dest='debug', action='store_true')
    parser.add_argument('--tta', help='apply test-time augmentations', dest='tta', action='store_true')
    parser.add_argument('--model-only', help='compose model only and output its description',
                        dest='model_only', action='store_true')
    parser.add_argument('--model-type',
                        help='model type for the TCC classification: either PC for pure classification or OR for ordinal regression',
                        dest='model_type', choices=['PC', 'OR', 'ORbin'])

    parser.add_argument('--serial', help='disable parallel queued data preprocessing and transfer to GPU, make it serial',
                        dest='serial', action='store_true')

    parser.add_argument('--residual', help='create residual architechture of the network. ' +
                                           'In this case, you will need to specify the number of residual blocks' +
                                           ' using the --blocks-num option', action='store_true')
    parser.add_argument('--blocks-num', help='number of convolutional blocks (three resnet blocks each) of the model',
                        dest='blocks_num', type=int, default=5)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot', help='Resume training from a snapshot.')

    group_arch = parser.add_mutually_exclusive_group()


    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):

    return parsed_args