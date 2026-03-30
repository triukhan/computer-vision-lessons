# model.py
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
