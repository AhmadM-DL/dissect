# -*- coding: utf-8 -*-
"""
Created on Tuesday April 14 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)

This module implements a number of standard benchmarks architectures, however by utilizing
DeepClusteringNet class.
"""
import torch.nn as nn
import torch

import math
import time
import os

import numpy as np

from custom_layers import SobelFilter
from deep_clustering_net import DeepClusteringNet
from sklearn.metrics import normalized_mutual_info_score

from  layers_stacker import stack_convolutional_layers, stack_linear_layers

def AlexNet_Micro(sobel, batch_normalization, device, concat_sobel=False):
    
    n_input_channels = 2 + int(not sobel) if not concat_sobel else 5

    alexnet_features_cfg = [
                {
                "type": "convolution",
                "out_channels":64,
                "kernel_size":3,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":2,
                "stride":None,
                },

                {
                "type": "convolution",
                "out_channels":192,
                "kernel_size":3,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":2,
                "stride":None,
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },
                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                }         
                ]

    classifier_cfg = [
                      {"type":"drop_out",
                       "drop_ratio": 0.6},

                      {"type":"linear",
                       "out_features":2048,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.6},

                      {"type":"linear",
                      "out_features":2048}
        ]

    model = DeepClusteringNet(input_size=(3,32,32),
                              features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                              classifier= stack_linear_layers(input_features= 256 * 4 * 4, cfg= classifier_cfg),
                              top_layer = None,
                              with_sobel=sobel,
                              device=device)
    return model

def AlexNet_Small(sobel, batch_normalization, device, concat_sobel=False):
    """Implementation of AlexNet for CIFAR dataset

    Arguments:
        sobel {Boolean} -- Add sobel filter prior to input or not 
        batch_normalization {Boolean} -- Add normalization after convolution or not
        device {torch.device} -- Pytorch device to send the model to (cpu/gpu)
    """
    n_input_channels = 2 + int(not sobel) if not concat_sobel else 5
    
    alexnet_features_cfg = [
                {
                "type": "convolution",
                "out_channels":64,
                "kernel_size":11,
                "stride":4,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":192,
                "kernel_size":5,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },
                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                }         
                ]

    classifier_cfg = [
                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                       "out_features":2048,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                      "out_features":2048}
        ]

    model = DeepClusteringNet(input_size=(3,224,224),
                              features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                              classifier= stack_linear_layers(input_features= 256 * 6 * 6, cfg= classifier_cfg),
                              top_layer = None,
                              with_sobel=sobel,
                              device=device)
    return model

def AlexNet_ImageNet(sobel, batch_normalization, device, concat_sobel=False):
    """Implementation of AlexNet for Imagenet dataset

    Arguments:
        sobel {Boolean} -- Add sobel filter prior to input or not 
        batch_normalization {Boolean} -- Add normalization after convolution or not
        device {torch.device} -- Pytorch device to send the model to (cpu/gpu)
    """
    n_input_channels = 2 + int(not sobel) if not concat_sobel else 5
    
    alexnet_features_cfg = [
                {
                "type": "convolution",
                "out_channels":96,
                "kernel_size":11,
                "stride":4,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":5,
                "stride":1,
                "padding":2,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":384,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type": "convolution",
                "out_channels":256,
                "kernel_size":3,
                "stride":1,
                "padding":1,
                "activation":"ReLU",
                },

                {
                "type":"max_pool",
                "kernel_size":3,
                "stride":2,
                }         
                ]

    classifier_cfg = [
                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                       "out_features":4096,
                       "activation":"ReLU"},

                      {"type":"drop_out",
                       "drop_ratio": 0.5},

                      {"type":"linear",
                      "out_features":4096}
        ]

    model = DeepClusteringNet(input_size=(3,224,224),
                              features= stack_convolutional_layers(input_channels= n_input_channels, cfg=alexnet_features_cfg, batch_normalization=batch_normalization),
                              classifier= stack_linear_layers(input_features= 256 * 6 * 6, cfg= classifier_cfg),
                              top_layer = None,
                              with_sobel=sobel,
                              device=device)
    return model

def LeNet(batch_normalization, device):
    """
    Implementation of LeNet
    """
    lenet_features_cfg = [{"type":"convolution",
                            "out_channels": 6,
                            "kernel_size": 5,
                            "padding": 0,
                            "stride":1,
                            "activation": "ReLU",
                            },
                            {"type":"max_pool",
                            "kernel_size": 2,
                            "stride": 2,
                            },
                            {"type":"convolution",
                            "out_channels": 16,
                            "kernel_size": 5,
                            "padding": 0,
                            "stride":1,
                            "activation": "ReLU",
                            },
                            {"type":"max_pool",
                            "kernel_size": 2,
                            "stride": 2,
                            }]

    classifier_cfg = [{"type":"linear", "out_features": 120, "activation":"ReLU"},
                      {"type":"linear", "out_features": 84 }]

    model = DeepClusteringNet(
                       input_size=(1,32,32),
                       features= stack_convolutional_layers(input_channels=1, cfg=lenet_features_cfg, batch_normalization=batch_normalization),
                       classifier= stack_linear_layers(input_features=16*5*5, cfg= classifier_cfg),
                       top_layer= None,
                       with_sobel=False,
                       device=device)

    return model