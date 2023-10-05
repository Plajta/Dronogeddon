from read_dataset import batch_size

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
LOGGING = True

class Universal(nn.Module):
    def __init__(self):
        super(Universal, self).__init__()
        self.model_iter = 0
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy"
        }
    
    def set_model_to_trainable(self, freeze):
        if freeze == "half-freeze":
            for name, param in self.model.named_parameters():
                if not ("layer4" in name or "fc" in name): #other layers that "layer4" or fc
                    param.requires_grad = False
                else: 
                    param.requires_grad = True


        elif freeze == "full-freeze":
            for name, param in self.model.named_parameters():
                param.requires_grad = False

    def train_net(self, train):
        idx = 0
        total_loss = 0
        total_loss_cls = 0
        total_loss_bbox = 0

        correct = 0

        self.train()
        self.set_model_to_trainable("half-freeze")
        for X, y in train:
            X = X.to(DEVICE)

            self.optimizer.zero_grad()
            output_bbox, output_cls = self(X)

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
            total_loss_cls += loss_cls.detach().item()
            total_loss_bbox += loss_bbox.detach().item()

            for i in range(output_cls.shape[0]):
                output_idx = torch.argmax(output_cls[i])
                target_idx = torch.argmax(target_cls[i])
                if output_idx == target_idx:
                    correct += 1
            
            #if idx % LOG_FREQ == 0:
            #    print("train loss: " + str(loss.item()))
            #if LOGGING:
            #    Wandb.wandb.log({"train/loss": loss.item(), "train/acc": correct / (idx * BATCH)})

                #logging losses separately
            #    Wandb.wandb.log({"train/loss_bbox": loss_bbox.item(), "train/loss_cls": loss_cls.item()})
                
            #free the memory
            torch.cuda.empty_cache()

        if LOGGING:
            Wandb.wandb.log({"train/loss": total_loss / idx, "train/acc": correct / (idx * BATCH)})
            Wandb.wandb.log({"train/loss_cls": total_loss_cls / idx, "train/loss_bbox": total_loss_bbox / idx})
    
    def test_net(self, test):
        self.eval()
        torch.no_grad()
        self.set_model_to_trainable("full-freeze")

        idx = 0
        correct = 0
        total_loss = 0
        total_loss_bbox = 0
        total_loss_cls = 0

        for X, y in test:
            X = X.to(DEVICE)

            output_bbox, output_cls = self(X)

            output_cls.to(DEVICE)
            output_bbox.to(DEVICE)

            target_cls = y[:, 0:3].to(DEVICE)
            target_bbox = y[:, 3:7].to(DEVICE)

            loss_cls = F.cross_entropy(output_cls, target_cls)
            loss_bbox = F.smooth_l1_loss(output_bbox, target_bbox) * 100 #scaling by 100 #TODO: check if correct

            loss = loss_cls + loss_bbox

            total_loss += loss.detach().item()
            total_loss_cls += loss_cls.detach().item()
            total_loss_bbox += loss_bbox.detach().item()

            idx += 1

            for i in range(output_cls.shape[0]):
                output_idx = torch.argmax(output_cls[i])
                target_idx = torch.argmax(target_cls[i])
                if output_idx == target_idx:
                    correct += 1

            print("test loss: " + str(loss.item()), "test acc: " + str(correct / (idx * BATCH)))
            #if LOGGING:
            #    Wandb.wandb.log({"test/loss": loss.item(), "test/acc": correct / (idx * BATCH)})

                #logging losses separately
            #    Wandb.wandb.log({"test/loss_bbox": loss_bbox.item(), "test/loss_cls": loss_cls.item()})

            #free the memory
            torch.cuda.empty_cache()

        if LOGGING:
            Wandb.wandb.log({"test/loss": total_loss / idx, "test/acc": correct / (idx * BATCH)})
            Wandb.wandb.log({"test/loss_cls": total_loss_cls / idx, "test/loss_bbox": total_loss_bbox / idx})

    def validate_net(self, valid):
        pass

    def run(self, model_name, train_loader, test_loader, valid_loader):

        print(DEVICE)
        print("Logging set to " + str(LOGGING))

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        #get info
        #torchinfo.summary(self, (1, 3, 640, 480))

        self.to(DEVICE)

        if LOGGING:
            Wandb.Init("DOOR-RCNN", self.config, model_name + " run:" + str(self.model_iter))
        
        self.test_net(test_loader)
        for i, epoch in enumerate(range(self.config["epochs"])):
            print(str(i) + " epoch")

            #train epoch and log
            self.train_net(train_loader)

            #test epoch and log
            self.test_net(test_loader)

            #validate after every epoch
            self.validate_net(valid_loader)

            torch.save(self, os.getcwd() + "/src/neural/saved_models/" + str(self.model_iter) + str(epoch) + ".pth")     
            torch.cuda.empty_cache()

        self.model_iter += 1
        if LOGGING:
            Wandb.End()

###################
# DoorCNN
###################

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

###################
# MyDoorResNet
###################

def res_block(in_c):
    return nn.Sequential(
        nn.Conv2d(in_c[0], in_c[1], (3, 3)),
        nn.Conv2d(in_c[1], in_c[2], (3, 3)),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(in_c[2])
    )

class MyDoorResNet(Universal):
    def __init__(self):
        super(MyDoorResNet, self).__init__()

        #convolutional
        self.block1 = res_block([3, 16, 32])
        self.pool1 = nn.MaxPool2d(2) #used in skip connection

        self.block1 = res_block([32, 64, 128])
        self.pool1 = nn.MaxPool2d(2)

        self.block1 = res_block([128, 256, 512])
        self.pool1 = nn.MaxPool2d(2)

        self.block1 = res_block([512, 1024, 2048])
        self.pool1 = nn.MaxPool2d(2)

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
        x_res = self.pool1(x)
        x = self.block1(x)
        x = x + x_res

        x_res = self.pool2(x)
        x = self.block2(x)
        x = x + x_res

        x_res = self.pool3(x)
        x = self.block3(x)
        x = x + x_res

        x_res = self.pool4(x)
        x = self.block4(x)
        x = x + x_res

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

        return pred_bbox, pred_cls


class DoorFC(nn.Module):
    def __init__(self, in_features):
        super(DoorFC, self).__init__()

        self.bbox = nn.Sequential(nn.Linear(in_features, 4), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(512, 3))

    def forward(self, x):
        return self.bbox(x), self.classifier(x)


class DoorResNet(Universal):
    def __init__(self):
        super(DoorResNet, self).__init__()

        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)

        num_features = self.model.fc.in_features
        FC_append = DoorFC(num_features)
        self.model.fc = FC_append

        self.set_model_to_trainable("half-freeze") #just to see number of parameters
        torchinfo.summary(self.model, (1, 3, 640, 480))

    def forward(self, x):
        pred = self.model(x)

        return pred[0], pred[1]