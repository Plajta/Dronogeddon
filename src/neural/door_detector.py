#TODO: list for this model
#1) add image pixel normalization to read_dataset - already done
#2) add logs to wandb - done
#3) optimize

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

import cv2
import numpy as np
import gc

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG_FREQ = 5
BATCH = batch_size

print(DEVICE)

#free allocated cuda memory
torch.cuda.empty_cache()
gc.collect()

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
            "metric": "accuracy"
        }

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

            for i, elem in enumerate(output_cls):
                if torch.equal(elem, target_cls[i]):
                    correct += 1
            
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

            for i, elem in enumerate(output_cls):
                if torch.equal(elem, target_cls[i]):
                    correct += 1

            idx += 1
            if idx % LOG_FREQ == 0:
                print("test loss: " + str(loss.item()), "test acc: " + str(correct / (idx * BATCH)))
                Wandb.wandb.log({"test_loss": loss.item(), "test_acc": correct / (idx * BATCH)})

            #free the memory
            torch.cuda.empty_cache()

        return total_loss / idx, correct / (idx * BATCH)

    def run(self, train_loader, test_loader, valid_loader):

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        #get info
        torchinfo.summary(self, (1, 3, 640, 480))

        self.to(DEVICE)

        Wandb.Init("DOOR-RCNN", self.config, "DOOR-CNN run:" + str(self.model_iter))
        
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
        for i in range(self.config["epochs"]):

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

            torch.save(self, os.getcwd() + "/src/neural/saved_models/" + str(self.model_iter) + str(i) + ".pth")

        self.model_iter += 1
        Wandb.End()

def run_model():
    #DoorModel = DoorRCNN()
    #DoorModel.run(train, test, validation)

    MyDoorModel = DoorCNN().to(DEVICE)
    MyDoorModel.run(train, test, validation)

def model_inference(path):
    #inference on notebook camera

    model = torch.load(os.getcwd() + "/src/neural/saved_models/" + path)

    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()

        transposed_array = frame.transpose((2, 1, 0))
        tensor = torch.tensor(transposed_array).unsqueeze(0).to(torch.float32).to(DEVICE)
        out = model(tensor)

        out_cls = out[0]
        out_bbox = out[1]

        idx = torch.where(out_cls == 1.0000)
        out_bbox = out_bbox[0].detach().cpu().numpy()

        cls = ""
        if idx == 0:
            cls = "closed"
        elif idx == 0:
            cls = "half open"
        else: #cls == "2"
            cls = "fully open"

        x_center = round(float(out_bbox[0]) * frame.shape[1])
        y_center = round(float(out_bbox[1]) * frame.shape[0])
        width = round(float(out_bbox[2]) * frame.shape[1])
        height = round(float(out_bbox[3]) * frame.shape[0])

        start_x = x_center - round(width / 2)
        start_y = y_center - round(height / 2)

        end_x = start_x + width
        end_y = start_y + height

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 5)
        cv2.putText(frame, cls, (start_x - 5, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        print(cls, start_x, start_y, end_x, end_y)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) &0xFF == ord("q"):
            break
    
    vid.release()
    cv2.destroyAllWindows()

#run_model()
model_inference("02.pth")