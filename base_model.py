from torch import nn
from monai.networks.nets import EfficientNetBN

class CVCModel(nn.Module):
    def __init__(self,pretrained = True):
        super(CVCModel,self).__init__()
        
        self.model = EfficientNetBN("efficientnet-b7",num_classes = 10,pretrained=pretrained)
        self.model._fc = nn.Sequential(
                nn.LazyLinear(512),
                nn.PReLU(),
                nn.Dropout(0.3),
                nn.Linear(512,256),
                nn.PReLU(),
                nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(10)
        )
        
    def forward(self,x):
        return self.model(x)
    
