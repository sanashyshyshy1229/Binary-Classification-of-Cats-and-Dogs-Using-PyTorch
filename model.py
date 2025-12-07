import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 加载预训练 ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 替换掉最后一层 FC，使其输出 2 分类
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)
