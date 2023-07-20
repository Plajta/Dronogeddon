from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

import os
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

transform = transforms.Compose([
    transforms.ToTensor()
])

class DoorsDataset(Dataset):
    def __init__(self, type):
        dataset_path = os.getcwd() + "/src/neural/dataset/"

        if type == "test":
            dataset_path += "test/"
        elif type == "valid":
            dataset_path += "valid/"
        else:
            dataset_path += "train/"

        self.img_labels, self.images = self.read_data(dataset_path + "labels/", dataset_path + "images/")

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        X = self.img_labels[index]
        y = self.images[index]
        return X, y

    def read_data(self, path_label, path_img):
        output_labels = []
        output_images = []

        for filename in os.listdir(path_label):

            #read label
            label_path = os.path.join(path_label, filename)
            file = open(label_path, "r")
            file_data = file.readline()

            data = file_data.split(" ")
            output_labels.append(data)

            file.close()

            #read image
            img_path = os.path.join(path_img, filename.replace(".txt", ".jpg"))
            img = read_image(img_path)
            
            #convert to size 640x480
            R = img[0, :, :].detach().cpu().numpy()
            G = img[1, :, :].detach().cpu().numpy()
            B = img[2, :, :].detach().cpu().numpy()

            R = cv2.resize(R, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            G = cv2.resize(G, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            B = cv2.resize(B, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
            
            img_resized = cv2.merge((R, G, B))
            tensor_resized = transform(img_resized)

            output_images.append(tensor_resized)

        return output_labels, output_images
    
def inspect_dataset(index, dataloader):
    label = dataloader.dataset[index][0]
    img = dataloader.dataset[index][1]

    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

train_data = DoorsDataset("train")
test_data = DoorsDataset("test")
valid_data = DoorsDataset("valid")

train = DataLoader(train_data, batch_size=32, shuffle=True)
test = DataLoader(test_data, batch_size=32, shuffle=True)
validation = DataLoader(valid_data, batch_size=32, shuffle=True)

#inspect_dataset(5, train)