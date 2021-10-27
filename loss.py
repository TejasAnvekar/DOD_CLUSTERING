import torch.nn as nn
from pytorch_msssim import SSIM


class loss():
    def __init__(self, device, loss="mse",channel=1):
        if loss == "mse":
            self.criterion = nn.MSELoss().to(device)
        if loss == "ssim":
            self.criterion = SSIM(data_range=1.0, channel=channel,
                                  size_average=True).to(device)
        if loss == "bce":
            self.criterion = nn.BCELoss().to(device)

    def __call__(self,p,gt):
        return self.criterion(p,gt)
