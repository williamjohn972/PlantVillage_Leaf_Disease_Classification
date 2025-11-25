import torch
import torch.nn as nn 
from torchvision import models

def get_resnet18(num_classes, dropout=0.0):

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # We need to remove the final classifier layer 
    # and replace it with our own 
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes))
    
    return model


