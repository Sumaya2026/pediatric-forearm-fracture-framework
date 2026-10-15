import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights



class URFNet(nn.Module):
    def __init__(self, num_classes=13):
        super(URFNet, self).__init__()

        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  

        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x1, x2):
 
        features1 = self.resnet(x1)
        features2 = self.resnet(x2)


        features = torch.stack([features1, features2], dim=1)  # Shape: [batch_size, 2, feature_dim]
        
        attn_output, _ = self.cross_attention(features, features, features)  # Cross-modal attention

        x = attn_output.mean(dim=1)  

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        output = self.classifier(x)
        return output



