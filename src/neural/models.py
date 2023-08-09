from read_dataset import train, test, validation, batch_size

from torchvision.models import resnet18, ResNet18_Weights #fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo
import os
import Wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG_FREQ = 5
BATCH = batch_size

class Universal(nn.Module):
    def __init__(self):
        super(Universal, self).__init__()
        self.model_iter = 0
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy"
        }
    
    def train_net(self, train):
        idx = 0
        total_loss = 0

        self.train()
        for X, y in train:
            X = X.to(DEVICE)

            self.optimizer.zero_grad()
            output_cls, output_bbox = self(X)

            output_bbox.to(DEVICE)
            output_cls.to(DEVICE)

            output_cls.to(DEVICE)
            output_bbox.to(DEVICE)

            target_cls = y[:, 0:3].to(DEVICE)
            target_bbox = y[:, 3:7].to(DEVICE)

            loss_cls = F.cross_entropy(output_cls, target_cls)
            loss_bbox = F.smooth_l1_loss(output_bbox, target_bbox) * 100 #scaling by 100 #TODO: check if correct

            loss = loss_cls + loss_bbox

            loss.backward()
            self.optimizer.step()

            idx += 1
            total_loss += loss.detach().item()

            pred_cls = output_cls.data.max(1, keepdim=True)
            correct += pred_cls.eq(target_cls.data.view_as(pred_cls).sum())
            
            if idx % LOG_FREQ == 0:
                print("train loss: " + str(loss.item()))
                Wandb.wandb.log({"train_loss": loss.item()})
                
            #free the memory
            torch.cuda.empty_cache()

        return total_loss / idx
    
    def test_net(self, test):
        self.eval()
        idx = 0
        correct = 0
        total_loss = 0

        for X, y in test:
            X = X.to(DEVICE)

            output_cls, output_bbox = self(X)

            output_cls.to(DEVICE)
            output_bbox.to(DEVICE)

            target_cls = y[:, 0:3].to(DEVICE)
            target_bbox = y[:, 3:7].to(DEVICE)

            loss_cls = F.cross_entropy(output_cls, target_cls)
            loss_bbox = F.smooth_l1_loss(output_bbox, target_bbox) * 100 #scaling by 100 #TODO: check if correct

            loss = loss_cls + loss_bbox

            total_loss += loss.detach().item()

            idx += 1

            pred_cls = output_cls.data.max(1, keepdim=True)
            correct += pred_cls.eq(target_cls.data.view_as(pred_cls).sum())
            #TODO: dodělat accuracy (je to napsaný v inferenci)

            #test set has only 286 images, so i am going to log this every iteration
            print("test loss: " + str(loss.item()), "test acc: " + str(correct / (idx * BATCH)))
            Wandb.wandb.log({"test_loss": loss.item(), "test_acc": correct / (idx * BATCH)})

            #free the memory
            torch.cuda.empty_cache()

        return total_loss / idx, correct / (idx * BATCH)

    def run(self, model_name, train_loader, test_loader, valid_loader):

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        #get info
        torchinfo.summary(self, (1, 3, 640, 480))

        self.to(DEVICE)

        Wandb.Init("DOOR-RCNN", self.config, model_name + " run:" + str(self.model_iter))
        
        init_loss, init_acc = self.test_net(test_loader)
        print(init_loss, init_acc)
        Wandb.wandb.log({"test_acc": init_acc, "test_loss": init_loss})
        for i, epoch in enumerate(range(self.config["epochs"])):

            #train epoch and log
            train_loss = self.train_net(train_loader)
            print(train_loss)
            print("train " + str(i) + " epoch")
            Wandb.wandb.log({"train_loss": train_loss})

            
            #test epoch and log
            test_loss, test_acc = self.test_net(test_loader)
            print(test_loss, test_acc)
            print("test " + str(i) + "epoch")
            Wandb.wandb.log({"test_acc": test_acc, "test_loss": test_loss})

            print("train loss:", train_loss)
            print("test loss:", test_loss, "test acc:", test_acc)

            torch.save(self, os.getcwd() + "/src/neural/saved_models/" + str(self.model_iter) + str(epoch) + ".pth")     

        self.model_iter += 1
        Wandb.End()

class DoorCNN(Universal):
    def __init__(self):
        super(DoorCNN, self).__init__()

        #convolutional
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.pool1 = nn.AvgPool2d(2)
        self.b_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 128, (3, 3))
        self.pool2 = nn.AvgPool2d(2)
        self.b_norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, (3, 3))
        self.pool3 = nn.AvgPool2d(2)
        self.b_norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, (3, 3))
        self.pool4 = nn.MaxPool2d(2)
        self.b_norm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 256, (3, 3))
        self.pool5 = nn.MaxPool2d(2)
        self.b_norm5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 64, (3, 3))
        self.pool6 =nn.MaxPool2d(2)
        self.b_norm6 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        #fully connected - classification
        self.linear1 = nn.Linear(2560, 512)
        self.drop1 = nn.Dropout(0.5)
        self.b_norm_l1 = nn.BatchNorm1d(512)

        self.linear2 = nn.Linear(512, 64)
        self.drop2 = nn.Dropout(0.5)
        self.b_norm_l2 = nn.BatchNorm1d(64)

        self.linear3 = nn.Linear(64, 16)
        self.drop3 = nn.Dropout(0.5)
        self.b_norm_l3 = nn.BatchNorm1d(16)

        self.linear_fin = nn.Linear(16, 3)

        #fully connected - bbox prediction
        self.linear1_bbox = nn.Linear(2560, 512)
        self.b_norm_bbox_l1 = nn.BatchNorm1d(512)

        self.linear2_bbox = nn.Linear(512, 16)
        self.b_norm_bbox_l2 = nn.BatchNorm1d(16)

        self.linear_bbox_fin = nn.Linear(16, 4)

    def forward(self, x):
        #convolution
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.b_norm1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.b_norm2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = self.b_norm3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(x)
        x = self.b_norm4(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = F.relu(x)
        x = self.b_norm5(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = F.relu(x)
        x = self.b_norm6(x)

        x = self.flatten(x)

        #fully connected - bbox prediction
        x_bbox = self.linear1_bbox(x)
        
        #fully-connected - classification
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

        pred_cls = self.linear_fin(x)

        #fully connected - bbox prediction
        x_bbox = self.b_norm_bbox_l1(x_bbox)

        x_bbox = self.linear2_bbox(x_bbox)
        x_bbox = self.b_norm_bbox_l2(x_bbox)

        x_bbox = self.linear_bbox_fin(x_bbox)

        pred_bbox = F.sigmoid(x_bbox)

        return pred_cls, pred_bbox

class MyDoorResNet(Universal):
    def __init__(self):
        super(MyDoorResNet, self).__init__()

        #convolutional
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.pool1 = nn.AvgPool2d(2)
        self.b_norm1 = nn.BatchNorm2d(BATCH)

        self.conv2 = nn.Conv2d(32, 128, (3, 3))
        self.pool2 = nn.AvgPool2d(2)
        self.b_norm2 = nn.BatchNorm2d(BATCH)

        self.conv3 = nn.Conv2d(128, 256, (3, 3))
        self.pool3 = nn.AvgPool2d(2)
        self.b_norm3 = nn.BatchNorm2d(BATCH)

        self.conv4 = nn.Conv2d(256, 512, (3, 3))
        self.pool4 = nn.MaxPool2d(2)
        self.b_norm4 = nn.BatchNorm2d(BATCH)

        self.conv5 = nn.Conv2d(512, 256, (3, 3))
        self.pool5 = nn.MaxPool2d(2)
        self.b_norm5 = nn.BatchNorm2d(BATCH)

        self.conv6 = nn.Conv2d(256, 64, (3, 3))
        self.pool6 =nn.MaxPool2d(2)
        self.b_norm6 = nn.BatchNorm2d(BATCH)

        self.flatten = nn.Flatten()

        #fully connected - classification
        self.linear1 = nn.Linear(2560, 512)
        self.drop1 = nn.Dropout(0.5)
        self.b_norm_l1 = nn.BatchNorm1d(512)

        self.linear2 = nn.Linear(512, 64)
        self.drop2 = nn.Dropout(0.5)
        self.b_norm_l2 = nn.BatchNorm1d(64)

        self.linear3 = nn.Linear(64, 16)
        self.drop3 = nn.Dropout(0.5)
        self.b_norm_l3 = nn.BatchNorm1d(16)

        self.linear_fin = nn.Linear(16, 3)

        #fully connected - bbox prediction
        self.linear1_bbox = nn.Linear(2560, 512)
        self.b_norm_bbox_l1 = nn.BatchNorm1d(512)

        self.linear2_bbox = nn.Linear(512, 16)
        self.b_norm_bbox_l2 = nn.BatchNorm1d(16)

        self.linear_bbox_fin = nn.Linear(16, 4)

    def forward(self, x):
        #convolution
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pool5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.pool6(x)
        x = F.relu(x)

        x = self.flatten(x)

        #fully connected - bbox prediction
        x_bbox = self.linear1_bbox(x)
        
        #fully-connected - classification
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

        pred_cls = self.linear_fin(x)

        #fully connected - bbox prediction
        x_bbox = self.b_norm_bbox_l1(x_bbox)

        x_bbox = self.linear2_bbox(x_bbox)
        x_bbox = self.b_norm_bbox_l2(x_bbox)

        x_bbox = self.linear_bbox_fin(x_bbox)

        pred_bbox = F.sigmoid(x_bbox)

        return pred_cls, pred_bbox


class DoorResNet(Universal):
    def __init__(self):
        super(DoorResNet, self).__init__()