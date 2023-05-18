from torchvision.models import detection
from torchvision import transforms
import torch
import cv2

DEVICE = "cpu"
SCREEN_CENTER = (320, 240)

model = detection.ssdlite320_mobilenet_v3_large(pretrained=True, progress=True,
	num_classes=91, pretrained_backbone=True).to(DEVICE)
model.eval()

tensor_transform = transforms.ToTensor()

def convert_to_tensor(frame):
    torch_tensor = tensor_transform(frame)
    torch_tensor = torch.unsqueeze(torch_tensor, dim=0)

    return torch_tensor

def process_data(output, frame):
    output = output[0]
    detect_coords = []

    for i in range(len(output["boxes"])):
        confidence = output["scores"][i]
        label = output["labels"][i]

        if confidence >= 0.9 and label == 1: #only detect humans
            box = output["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            detect_coords.append([startX, startY, endX, endY])

    return detect_coords

def compute_dev(results, frame):
    curr_max_size = 0
    res_i = 0
    for i, result in enumerate(results):
        (startX, startY, endX, endY) = result
        res_size = abs(endX - startX) * abs(endY - startY)
        if curr_max_size < res_size:
            curr_max_size = res_size

        res_i = i


    (startX, startY, endX, endY) = results[res_i] #unpack right person to follow
    
    area = abs(startX - endX) * abs(startY - endY)
    area_max = (2 * SCREEN_CENTER[0]) * (2 * SCREEN_CENTER[1])
    
    area_used = round(area / area_max, 2)

    #compute center
    y, x = int(abs(endX - startX) / 2 + startX), int(abs(endY - startY) / 2 + startY)
    cv2.circle(frame, (y, x), 3, (0, 0, 255), 3)

    #compute deviation
    y_dev = y - SCREEN_CENTER[0]
    x_dev = x - SCREEN_CENTER[1]

    return y_dev, x_dev, area_used