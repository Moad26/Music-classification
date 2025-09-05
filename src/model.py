from typing import Optional

import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=1,
        kernel_size=3,
        bias=False,
    )


def conv7x7(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=3,
        kernel_size=7,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
        padding=0,
        kernel_size=1,
        bias=False,
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(
            in_channels=out_channels, out_channels=out_channels, stride=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(
                    in_channels=in_channels, out_channels=out_channels, stride=stride
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return self.dropout(x)


class CNN_Music_classifier(nn.Module):
    def __init__(
        self, num_channels: int = 1, num_classes: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.stem = nn.Sequential(
            conv7x7(num_channels, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
        )
        self.layer1 = self._make_layer(
            in_channels=64, out_channels=64, num_blocks=2, first_stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=64, out_channels=128, num_blocks=2, first_stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=128, out_channels=256, num_blocks=2, first_stride=2
        )
        self.layer4 = self._make_layer(
            in_channels=256, out_channels=512, num_blocks=2, first_stride=2
        )
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
        self._init_layer()

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(
                    m.weight
                )  # It's the final linear layer ,so we used xavier initialisation
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        first_stride: int,
        num_blocks: int,
    ):
        layers = [
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride,
                dropout=self.dropout,
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=self.dropout,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
