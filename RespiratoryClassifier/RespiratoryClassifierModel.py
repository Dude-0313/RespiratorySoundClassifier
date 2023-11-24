# Description :
# Date : 11/17/2023 (17)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#
# Revisions :
#
import torch
import torch.nn as nn


class RespiratoryClassifierModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(RespiratoryClassifierModel, self).__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * ((input_shape[1] // 2)) * ((input_shape[2] // 2)), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.pipeline(x)


import torchvision.models as model_zoo


class RespiratoryClassifierDenseNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(RespiratoryClassifierDenseNet, self).__init__()
        self.model = model_zoo.densenet121(pretrained=True, progress=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_all(self):
        # Un-Freeze the weights of the network
        for param in self.model.parameters():
            param.requires_grad = True
