from torchvision import models
import torch.nn as nn


class CustomConvNet(nn.Module):
    def __init__(self, class_size, pretrained=True):
        super(CustomConvNet, self).__init__()

        self.class_size = class_size
        if pretrained:
            self.backbone = models.convnext_base(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.convnext_base(weights=None)
        
        self.backbone.classifier[2] = nn.Linear(1024, 512)
        self.projection_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        logit = self.projection_layer(x)
        return logit
    
class CustomEfficientNet(nn.Module):
    def __init__(self, class_size):
        super(CustomEfficientNet, self).__init__()

        self.class_size = class_size
        self.backbone = models.efficientnet_v2_m(weights='EfficientNet_V2_M_Weights.DEFAULT')
        # print(self.backbone)
        self.backbone.classifier[1] = nn.Linear(1280, 512)
        self.projection_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        logit = self.projection_layer(x)
        return logit
    

class CustomVIT(nn.Module):
    def __init__(self, class_size):
        super(CustomVIT, self).__init__()

        self.class_size = class_size
        self.backbone = models.vit_b_16(weights='IMAGENET1K_V1')
        self.backbone.heads.head = nn.Linear(768, 512)

        self.projection_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_size)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        logit = self.projection_layer(x)
        return logit

class CustomResNet(nn.Module):
    def __init__(self, class_size):
        super(CustomResNet, self).__init__()

        self.class_size = class_size
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(2048, 512)

        self.projection_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_size)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        logit = self.projection_layer(x)
        return logit