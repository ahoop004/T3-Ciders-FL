import torch
import torch.nn as nn
import torchvision.models as tv_models

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential( 
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                  
        )
        self.layer2 = nn.Sequential(
            # Use a smaller kernel to preserve dimensions
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                   
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)                   
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)  
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class MobileNetV2Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        backbone = tv_models.mobilenet_v2(weights='DEFAULT')
        
        in_feats = backbone.classifier[-1].in_features

        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        
        self.v2model = backbone

    def forward(self, x):
        return self.v2model(x)


        
class MobileNetV3Transfer(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 10):
        super().__init__()

        backbone = tv_models.mobilenet_v3_small(weights='DEFAULT')
        
        in_feats = backbone.classifier[-1].in_features

        backbone.classifier[-1] = nn.Linear(in_feats, num_classes)
        
        self.v3model = backbone

    def forward(self, x):
        return self.v3model(x)



        