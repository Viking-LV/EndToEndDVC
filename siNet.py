# from __future__ import division
# import tensorflow as tf
# # import tensorflow.contrib.slim as slim
# import tf_slim as slim
# import numpy as np
# import torch
# from tensorflow.contrib.layers.python.layers import initializers

import torch.utils.data
from torch.nn import functional as F

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


# def nm(x):  # changed to None
#     w0=tf.Variable(1.0,name='w0')
#     w1=tf.Variable(0.0,name='w1')
#     return w0*x+w1*slim.batch_norm(x)


class SiNet(nn.Module):
    def __init__(self):
        super(SiNet, self).__init__()

        self.conv1 = Conv2d(in_channels=2, out_channels=32, kernel_size=3, dilation=1)

        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2)

        self.conv3 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=4)

        self.conv4 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=8)

        self.conv5 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=16)

        self.conv6 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=32)

        self.conv7 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=64)

        self.conv8 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=128)

        self.conv9 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1)


        self.conv10 = Conv2d(in_channels=32, out_channels=1, kernel_size=1, dilation=1)


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        return out


# def siNet(input):
#
#     dim = input.shape[1]
#
#     net1 = Conv2d(in_channels=dim, out_channels=32, kernel_size=3, dilation=1)
#     out1 = net1(input)
#     # net=slim.conv2d(input,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv1', data_format='NCHW')
#
#     net2 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=2)
#     out2 = net2(out1)
#     # net=slim.conv2d(net,32,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv2', data_format='NCHW')
#
#
#     net3 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=4)
#     out3 = net3(out2)
#     # net=slim.conv2d(net,32,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv3', data_format='NCHW')
#
#
#     net4 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=8)
#     out4 = net4(out3)
#     # net=slim.conv2d(net,32,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv4', data_format='NCHW')
#
#
#     net5 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=16)
#     out5 = net5(out4)
#     # net=slim.conv2d(net,32,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv5', data_format='NCHW')
#
#
#     net6 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=32)
#     out6 = net6(out5)
#     # net=slim.conv2d(net,32,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv6', data_format='NCHW')
#
#
#     net7 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=64)
#     out7 = net7(out6)
#     # net=slim.conv2d(net,32,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv7', data_format='NCHW')
#
#     net8 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=128)
#     out8 = net8(out7)
#     # net=slim.conv2d(net,32,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv8', data_format='NCHW')
#
#     net9 = Conv2d(in_channels=32, out_channels=32, kernel_size=3, dilation=1)
#     out9 = net9(out8)
#     # net=slim.conv2d(net,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv9', data_format='NCHW')
#
#     if dim == 1:
#         net10 = Conv2d(in_channels=32, out_channels=1, kernel_size=3, dilation=1)
#     else:
#         net10 = Conv2d(in_channels=32, out_channels=2, kernel_size=3, dilation=1)
#     out = net10(out9)
#     # net=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_conv_last', data_format='NCHW')
#     return out



# def siNet(input):
#
#     net=slim.conv2d(input,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv1', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv2', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv3', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv4', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv5', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv6', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv7', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv8', data_format='NCHW')
#     net=slim.conv2d(net,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv9', data_format='NCHW')
#     net=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_conv_last', data_format='NCHW')
#     return net