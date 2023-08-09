#TODO: list for this model
#1) add image pixel normalization to read_dataset - already done
#2) add logs to wandb - done
#3) optimize

from read_dataset import train, test, validation, batch_size
from models import DoorCNN, DoorResNet, MyDoorResNet

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

class DoorRCNN(nn.Module):
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

def run_models():
    #DoorModel = DoorRCNN()
    #DoorModel.run(train, test, validation)

    MyDoorModel = DoorCNN().to(DEVICE)
    MyDoorModel.run("DOOR-CNN", train, test, validation)

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

        idx = torch.argmax(out_cls)
        out_bbox = out_bbox[0].detach().cpu().numpy()

        cls = ""
        if idx == 0:
            cls = "closed"
        elif idx == 1:
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

run_models()
#model_inference("04.pth")