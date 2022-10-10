from torchvision import models
import torch.nn as nn
class ResNet50Fc(nn.Module):
    """
    ** input image should be in range of [0, 1]**
    """

    def __init__(self, num_classes, output_feature=False):
        super(ResNet50Fc, self).__init__()

        self.model_resnet = models.resnet50(pretrained=True)
        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features
        self.output_feature=output_feature
        self.bottleneck = nn.Linear(self.__in_features, 256)
        self.relu_bottle = nn.ReLU()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x= self.relu_bottle(x)
        c = self.fc(x)
        if self.output_feature:
            return x, c
        else:
            return c

    def output_num(self):
        return self.__in_features
