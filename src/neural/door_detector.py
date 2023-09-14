#TODO: list for this model
#1) add image pixel normalization to read_dataset - already done
#2) add logs to wandb - done
#3) optimize

from read_dataset import train, test, validation, batch_size
from models import DoorCNN, DoorResNet, MyDoorResNet, DEVICE
import models

from torchvision.models import resnet18, ResNet18_Weights #fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo
import os
import Wandb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

from torchvision.io import read_image

import cv2
import numpy as np
import gc

LOG_FREQ = 5
BATCH = batch_size


#free allocated cuda memory
torch.cuda.empty_cache()
gc.collect()

def run_models(wandb_logging, version):
    models.LOGGING = wandb_logging

    if version == 0:
        MyDoorModel = DoorCNN().to(DEVICE)
        MyDoorModel.run("DOOR-CNN", train, test, validation)
    elif version == 1:
        MyDoorModel = MyDoorResNet().to(DEVICE)
        MyDoorModel.run("DOOR-MyResNet", train, test, validation)
    elif version == 2:
        MyDoorModel = DoorResNet().to(DEVICE)
        MyDoorModel.set_model_to_trainable("half-freeze")
        MyDoorModel.run("DOOR-ResNet largeFC", train, test, validation)

def model_inference(path, type):
    #inference on notebook camera

    model = torch.load(os.getcwd() + "/src/neural/saved_models/" + path)


    if type == "camera":
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()

            transposed_array = frame.transpose((2, 1, 0))
            tensor = torch.tensor(transposed_array).unsqueeze(0).to(torch.float32).to(DEVICE)
            out = model(tensor)

            out_bbox = out[0]
            out_cls = out[1]

            idx = torch.argmax(out_cls)
            out_bbox = out_bbox[0].detach().cpu().numpy()

            print(out_bbox)

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
    elif type == "local":
        #local resources
        path_imgs = os.getcwd() + "/src/neural/dataset/valid/images/"
        arr_images = os.listdir(path_imgs)

        for file in arr_images:
            frame = read_image(path_imgs + file)
            frame_show = cv2.imread(path_imgs + file)

            tensor = torch.tensor(frame).unsqueeze(0).to(torch.float32).to(DEVICE)
            out = model(tensor)

            out_bbox = out[0]
            out_cls = out[1]

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

            cv2.rectangle(frame_show, (start_x, start_y), (end_x, end_y), (0, 255, 0), 5)
            cv2.putText(frame_show, cls, (start_x - 5, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(cls, start_x, start_y, end_x, end_y)

            cv2.imshow("frame", frame_show)

            cv2.waitKey(0)

run_models(True, 2)
#model_inference("05.pth", "local")