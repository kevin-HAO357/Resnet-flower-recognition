import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

class ResNet34(nn.Module):
    def __init__(self, num_classes=14):
        super(ResNet34, self).__init__() 

        # 确保了底层卷积核已经学会了提取通用视觉特征
        self.resnet34 = resnet34(weights=ResNet34_Weights.DEFAULT)

        # 替换全连接层
        # 新替换的全连接层默认 requires_grad = True，整个网络只有这一层在学习！
        self.resnet34.fc = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(self.resnet34.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet34(x)