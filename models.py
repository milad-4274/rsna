import torch
from torch import nn
import torchvision
from torchvision import datasets, models



class Resnet18Extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet18(pretrained=True)
        self.extractor = nn.Sequential(*list(model.children())[:-1])
        print(model)
    
    def forward(self,x):
        return self.extractor(x)

    def get_transform(self):
        return torchvision.transforms.transforms.Resize((224,224))
