from read_dataset import train, test, validation

from torchvision.models import resnet18, ResNet18_Weights #fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo
import os
import Wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "CPU"

class DoorFC(nn.Module):
    def __init__(self, in_features):
        super(DoorFC, self).__init__()

        self.bbox = nn.Sequential(nn.BatchNorm1d(512), nn.Dropout(), nn.Linear(in_features, 4))
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(), nn.Linear(128, 3))

    def forward(self, x):
        return self.bbox(x), self.classifier(x)

class DoorRCNN:
    def __init__(self):
        super(DoorRCNN, self).__init__()
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy",
            "log_freq": 100
        }

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

    def train_net(self):
        pass

    def test_net(self):
        pass

    def set_model_to_trainable(self):
        for param in self.model.parameters():
            param.requires_grad = False

        torchinfo.summary(self.model, (1, 3, 640, 480))

        num_features = self.model.fc.in_features
        FC_append = DoorFC(num_features)
        self.model.fc = FC_append
        
        torchinfo.summary(self.model, (1, 3, 640, 480))

    def forward(self, x):
        pred = self.model(x)

        return pred
    
    def run(self, train_loader, test_loader, valid_loader):
        self.set_model_to_trainable()

        #Wandb.Init("DOOR-RCNN", self.config, "DOOR-RCNN run:" + str(self.model_iter))
        
        #for i, epoch in enumerate(range(self.config["epochs"])):
            


DoorModel = DoorRCNN()
DoorModel.run(train, test, validation)