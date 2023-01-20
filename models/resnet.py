import torch
import torch.nn as nn

from blocks import ResnetBlock, StemBlock
from utils import conv1x1
from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple, Type


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        in_planes: int = 64,
        dilation: int = 1,
        image_shape: Tuple[int, int, int] = (3, 512, 512),
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_fn: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_fn is None:
            norm_layer = nn.ReLU(inplace=True)

        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = dilation
        self.inplanes = in_planes

        self.stem = StemBlock(input_channels=image_shape[0], output_channels=in_planes)
        self.num_of_layers = len(layers) + 1
        out_channels = in_planes
        for i, layer in enumerate(layers):
            if i == 0:
                setattr(self, f"conv{i+1}", self._make_layer(out_channels, layer))
            else:
                setattr(
                    self,
                    f"conv{i+1}",
                    self._make_layer(
                        out_channels,
                        layer,
                        stride=2,
                        dilate=replace_stride_with_dilation[i - 1],
                    ),
                )
            out_channels *= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResnetBlock.expansion, num_classes)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * ResnetBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * ResnetBlock.expansion, stride),
                self._norm_layer(planes * ResnetBlock.expansion),
            )

        layers = []
        layers.append(
            ResnetBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                self._norm_layer,
            )
        )
        self.inplanes = planes * ResnetBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                ResnetBlock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for i in range(1, self.num_of_layers):
            x = getattr(self, f"layer{i}")(x)
        return x


def _resnet(
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    return ResNet(layers=layers, **kwargs)


if __name__ == "__main__":
    arr = torch.ones(size=(4, 64, 100, 200))
    model = _resnet(layers=[3, 4, 6, 3])
    print(model)
