from read_dataset import train, test, validation

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchinfo
import os

class DoorRCNN:
    def __init__(self):
        super(DoorRCNN, self).__init__()
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy",
            "log_freq": 100
        }

        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        
        torchinfo.summary(self.model, (1, 3, 640, 480))

    def forward(self, x):
        pred = self.model(x)

        return pred
    
    def run(self, train_loader, test_loader, valid_loader):
        pass