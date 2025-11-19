import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU -> MaxPool"""

    def __init__(self, in_ch, out_ch, kernel_size=3, pool=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNNv2(nn.Module):
    """
    User 2 model (PlantVillage):
    - deeper than model_v1
    - uses LeakyReLU
    - includes dropout inside feature extractor
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),            # 224 -> 112
            ConvBlock(32, 64),           # 112 -> 56
            ConvBlock(64, 96),           # 56  -> 28
            ConvBlock(96, 128),          # 28  -> 14
            ConvBlock(128, 128, pool=False),  # keep 14x14
            nn.Dropout2d(0.3),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global avg pooling

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
