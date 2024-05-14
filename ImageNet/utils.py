import torch.nn as nn
import torch

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

    def forward(self, input):

        x = input.clone()
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.ms[0][i]) / self.ms[1][i]
        return x

class Interpolate(torch.nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = torch.nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
