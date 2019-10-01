import torch
from torch import nn 

class Model(nn.Module):
    def __init__(self, layer1, layer2):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, layer1),
            nn.ReLU(),
            nn.Linear(layer1, layer2),
            nn.ReLU(),
            nn.Linear(layer2,1)
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        return self.model(x)
