import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 256, 128]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x
