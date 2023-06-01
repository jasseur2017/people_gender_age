import torch
from torch import nn
from torchvision.ops import deform_conv2d
import math


class DCN(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        dilation=1
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        channels_ = 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels, channels_, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.init_offset()

    def init_offset(self,):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        offset_mask = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input, offset, self.weight, self.bias,
            self.stride, self.padding, self.dilation, mask
        )
