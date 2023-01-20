import torch.nn as nn

from torch import Tensor
from typing import Callable, Optional
from utils import conv1x1, conv3x3


class BasicBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        downsample: Optional[nn.Module] = None,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.skip = nn.Sequential()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(input_channels, output_channels, stride)
        self.batch_norm1 = norm_layer(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(output_channels, output_channels)
        self.batch_norm2 = norm_layer(output_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.batch_norm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        downsample: Optional[nn.Module] = None,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.conv1 = conv1x1(input_channels, input_channels, stride)
        self.batch_norm1 = norm_layer(input_channels)
        self.conv2 = conv3x3(input_channels, input_channels)
        self.batch_norm2 = norm_layer(input_channels)
        self.conv3 = conv1x1(input_channels, output_channels, stride)
        self.batch_norm3 = norm_layer(output_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class StemBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_fn: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_fn is None:
            activation_fn = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = norm_layer(output_channels)
        self.activation_fn = activation_fn
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = self.maxpool(x)
        return x
