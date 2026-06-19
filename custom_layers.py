import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):

        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        x = x * self.sigmoid(avg_out + max_out)

        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        x = x * self.sigmoid(
            self.conv_spatial(
                torch.cat([avg, max_], dim=1)
            )
        )

        return x


class ResCBAM(nn.Module):

    def __init__(self):
        super().__init__()
        self._built = False

    def _build(self, c):

        self.conv1 = nn.Conv2d(c, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

        self.conv3 = nn.Conv2d(c, c, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)

        self.cbam = CBAM(c)

        self.relu = nn.ReLU(inplace=True)

        self._built = True

    def forward(self, x):

        if not self._built:
            self._build(x.shape[1])

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.cbam(out)

        out += identity

        return self.relu(out)