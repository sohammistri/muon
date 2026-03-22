import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, channels=None, dropout=0.1):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout),
            ])
            c_in = c_out

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
