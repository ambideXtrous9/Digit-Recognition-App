
import torch
from torch import nn


class MNISTNeuralNet(nn.Module):
    def __init__(self,hidden_dim=512, dropout_prob = 0.2):
        super().__init__()

        self.conv = nn.Sequential( #1x28x28
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5), # 8x24x24
            nn.Dropout2d(p=dropout_prob,inplace=True),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.MaxPool2d(2), # 8x12x12
            nn.Conv2d(in_channels=8,out_channels=32,kernel_size=3), # 32x10x10
            nn.Dropout2d(p=0.4,inplace=True),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2), #32x5x5
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2), # 64x4x4
            nn.Dropout2d(p=dropout_prob,inplace=True),
            nn.BatchNorm2d(64)
            )

        self.linear = nn.Sequential(
            nn.Linear(in_features=64*4*4,out_features=hidden_dim),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Dropout(p=dropout_prob,inplace=True),
            nn.Linear(in_features=hidden_dim,out_features=10),
            nn.Softmax(dim=1))
        
        self._initialize_weights()  # weight initialization

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x