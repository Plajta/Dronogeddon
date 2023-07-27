from read_dataset import train, test, validation

from torchvision.models import resnet18, ResNet18_Weights #fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo
import os
import Wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

device = "CPU"

class DoorFC(nn.Module):
    def __init__(self, in_features):
        super(DoorFC, self).__init__()

        self.bbox = nn.Sequential(nn.BatchNorm1d(512), nn.Dropout(), nn.Linear(in_features, 4))
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(), nn.Linear(128, 3))

    def forward(self, x):
        return self.bbox(x), self.classifier(x)

class DoorCNN(nn.Module): #my own implementation :>
    def __init__(self):
        super(DoorCNN, self).__init__()
        self.model_iter = 0
        self.config = {
            "epochs": 20,
            "optimizer": "adam",
            "metric": "accuracy",
            "log_freq": 100
        }

        #convolutional
        self.conv1 = nn.Conv2d(3, 32, (3, 3), dilation=(5, 5))
        self.pool1 = nn.AvgPool2d(3)
        self.b_norm1 = nn.BatchNorm2d(32) #after ReLU

        self.conv2 = nn.Conv2d(32, 64, (3, 3), dilation=(3, 3))
        self.pool2 = nn.AvgPool2d(6)
        self.b_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.pool3 = nn.AvgPool2d(9)
        self.b_norm3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(128, 256, (9, 9))
        self.pool4 = nn.MaxPool2d(20)
        self.b_norm4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(256, 32, (3, 3))
        self.pool5 = nn.MaxPool2d(40)
        self.b_norm5 = nn.BatchNorm2d(32)

        #fully connected
        self.linear1 = nn.Linear(204281, 2048)
        self.drop1 = nn.Dropout(0.5)
        self.b_norm_l1 = nn.BatchNorm1d(32)

        self.linear2 = nn.Linear(2048, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.b_norm_l2 = nn.BatchNorm1d(32)

        self.linear3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(0.5)
        self.b_norm_l3 = nn.BatchNorm1d(32)

        self.linear4 = nn.Linear(512, 64)
        self.drop4 = nn.Dropout(0.5)
        self.b_norm_l4 = nn.BatchNorm1d(32)

        self.linear5 = nn.Linear(64, 16)
        self.drop5 = nn.Dropout(0.5)
        self.b_norm_l5 = nn.BatchNorm1d(32)

        self.linear_fin = nn.Linear(16, 3)

    def forward(self, x):
        #convolution
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = F.relu(x)
        x = self.drop5(x)

        #fully-connected
        x = self.linear1(x)
        x = self.drop1(x)
        x = F.relu(x)
        x = self.b_norm_l1(x)

        x = self.linear2(x)
        x = self.drop2(x)
        x = F.relu(x)
        x = self.b_norm_l2(x)

        x = self.linear3(x)
        x = self.drop3(x)
        x = F.relu(x)
        x = self.b_norm_l3(x)

        x = self.linear4(x)
        x = self.drop4(x)
        x = F.relu(x)
        x = self.b_norm_l4(x)

        x = self.linear5(x)
        x = self.drop5(x)
        x = F.relu(x)
        x = self.b_norm_l5(x)

        pred = self.linear_fin(x)
        return pred
    
    def train_net(self, train):
        self.model.train()
        for X, y in train:
            self.model(X)

    def test_net(self, test):
        pass

    def run(self, train_loader, test_loader, valid_loader):
        self.set_model_to_trainable()

        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        #test
        self.train_net(train_loader)

        Wandb.Init("DOOR-RCNN", self.config, "DOOR-RCNN run:" + str(self.model_iter))
        
        init_loss, init_acc = self.test_net(test_loader, Wandb.wandb)
        Wandb.wandb.log({"test_acc": init_acc, "test_loss": init_loss})
        for i, epoch in enumerate(range(self.config["epochs"])):

            #train epoch and log
            train_loss, train_acc = self.train_net(train_loader, Wandb.wandb)
            print("train " + str(i) + "epoch")
            Wandb.wandb.log({"train_acc": train_acc, "train_loss": train_loss})

            #test epoch and log
            test_loss, test_acc = self.test_net(test_loader, Wandb.wandb)
            print("test " + str(i) + "epoch")
            Wandb.wandb.log({"test_acc": test_acc, "test_loss": test_loss})

            print("train loss:", train_loss, "train acc:", train_acc)
            print("test loss:", test_loss, "test acc:", test_acc)

            torch.save(self, os.getcwd() + "/src/neural/saved_models/" + str(self.model_iter) + str(epoch) + ".pth")

        self.model_iter += 1
        Wandb.End()

class DoorRCNN:
    def __init__(self):
        super(DoorRCNN, self).__init__()
        self.model_iter = 0
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy",
            "log_freq": 100
        }

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

    def train_net(self, train):
        self.model.train()
        correct = 0
        log_i = 0

        for i, (data, target) in enumerate(train):
            log_i += 1
            self.optimizer.zero_grad()
            output_bbox, output_class = self.model(data)
            loss = nn.NLLLoss()

            print(target)


    def test_net(self, test):
        pass

    def set_model_to_trainable(self):
        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        FC_append = DoorFC(num_features)
        self.model.fc = FC_append

        torchinfo.summary(self.model, (1, 3, 640, 480))

    def forward(self, x):
        pred = self.model(x)

        return pred
    
    def run(self, train_loader, test_loader, valid_loader):
        self.set_model_to_trainable()

        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        #test
        self.train_net(train_loader)

        Wandb.Init("DOOR-RCNN", self.config, "DOOR-RCNN run:" + str(self.model_iter))
        
        init_loss, init_acc = self.test_net(test_loader, Wandb.wandb)
        Wandb.wandb.log({"test_acc": init_acc, "test_loss": init_loss})
        for i, epoch in enumerate(range(self.config["epochs"])):

            #train epoch and log
            train_loss, train_acc = self.train_net(train_loader, Wandb.wandb)
            print("train " + str(i) + "epoch")
            Wandb.wandb.log({"train_acc": train_acc, "train_loss": train_loss})

            #test epoch and log
            test_loss, test_acc = self.test_net(test_loader, Wandb.wandb)
            print("test " + str(i) + "epoch")
            Wandb.wandb.log({"test_acc": test_acc, "test_loss": test_loss})

            print("train loss:", train_loss, "train acc:", train_acc)
            print("test loss:", test_loss, "test acc:", test_acc)

            torch.save(self, os.getcwd() + "/src/neural/saved_models/" + str(self.model_iter) + str(epoch) + ".pth")

        self.model_iter += 1
        Wandb.End()

print("model init")

DoorModel = DoorRCNN()
DoorModel.run(train, test, validation)

MyDoorModel = DoorCNN()