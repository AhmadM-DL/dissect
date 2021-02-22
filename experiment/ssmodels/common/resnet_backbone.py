from torch import nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, architecture, path):
        super().__init__()

        self.model = models.__dict__[architecture](pretrained=False)
        del self.model.fc

        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x