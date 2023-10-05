from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import torch

import os
import torchvision.transforms as transforms
import cv2

transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 16 #i hate cuda-out of memory

def ImgTransform(img, justconvert):
    #convert to size 640x480
    R = img[0, :, :].detach().cpu().numpy()
    G = img[1, :, :].detach().cpu().numpy()
    B = img[2, :, :].detach().cpu().numpy()

    if justconvert: #to convert to numpy array
        return cv2.merge((B, G, R)) #for cv2 support

    R = cv2.resize(R, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    G = cv2.resize(G, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(B, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    
    img_resized = cv2.merge((R, G, B))
    tensor_resized = transform(img_resized)

    return tensor_resized

class DoorsDataset(Dataset):
    def __init__(self, type):
        self.dataset_path = os.getcwd() + "/src/neural/dataset/"

        #classes
        self.closed_num = 0
        self.fully_opened = 0
        self.half_opened = 0

        if type == "test":
            self.dataset_path += "test/"
        elif type == "valid":
            self.dataset_path += "valid/"
        else:
            self.dataset_path += "train/"

        self.path_labels = self.dataset_path + "labels/"
        self.path_imgs = self.dataset_path + "images"

        self.scan()

    def __len__(self):
        return len(os.listdir(self.path_labels))
    
    def __print_statistics__(self):
        print("n. o. closed: " + str(self.closed_num))
        print("n. o. fully opened " + str(self.fully_opened))
        print("n. o. half opened " + str(self.half_opened))

    def scan(self):
        arr_labels = os.listdir(self.path_labels)
        for label in arr_labels:
            label_path = os.path.join(self.path_labels, label)
            file = open(label_path, "r")
            file_data = file.readline()
            file.close()

            y = file_data.split(" ") #y - labels
            if int(y[0]) == 0:
                self.closed_num += 1
            elif int(y[0]) == 1:
                self.half_opened += 1
            elif int(y[0]) == 2:
                self.fully_opened += 2
    
    def __getitem__(self, index):

        arr_labels = os.listdir(self.path_labels)
        arr_images = os.listdir(self.path_imgs)

        label_path = os.path.join(self.path_labels, arr_labels[index])
        file = open(label_path, "r")
        file_data = file.readline()
        file.close()

        index_img = 0

        identify = label_path.replace(self.dataset_path + "labels/", "").replace(".txt", "")
        for i, image_path in enumerate(arr_images):
            if identify in image_path:
                index_img = i

        img_path = os.path.join(self.path_imgs, arr_images[index_img].replace(".txt", "jpg"))
        img = read_image(img_path)

        X = ImgTransform(img, False)
        y = file_data.split(" ") #y - labels

        cls = one_hot(torch.tensor(int(y[0])), num_classes=3)
        bbox = torch.tensor([float(x) for x in y[1:5]])
        y = torch.cat((cls, bbox))

        return X, y
    
class DetectDoorsDataset(Dataset): #TODO
    def __init__(self, type):
        self.dataset_path = os.getcwd() + "/src/neural/dataset/"

        #classes
        self.closed_num = 0
        self.fully_opened = 0
        self.half_opened = 0

        if type == "test":
            self.dataset_path += "test/"
        elif type == "valid":
            self.dataset_path += "valid/"
        else:
            self.dataset_path += "train/"

        self.path_labels = self.dataset_path + "labels/"
        self.path_imgs = self.dataset_path + "images"

        self.scan()

    def __len__(self):
        return len(os.listdir(self.path_labels))
    
    def __print_statistics__(self):
        print("n. o. closed: " + str(self.closed_num))
        print("n. o. fully opened " + str(self.fully_opened))
        print("n. o. half opened " + str(self.half_opened))

    def scan(self):
        arr_labels = os.listdir(self.path_labels)
        for label in arr_labels:
            label_path = os.path.join(self.path_labels, label)
            file = open(label_path, "r")
            file_data = file.readline()
            file.close()

            y = file_data.split(" ") #y - labels
            if int(y[0]) == 0:
                self.closed_num += 1
            elif int(y[0]) == 1:
                self.half_opened += 1
            elif int(y[0]) == 2:
                self.fully_opened += 2
    
    def __getitem__(self, index):

        arr_labels = os.listdir(self.path_labels)
        arr_images = os.listdir(self.path_imgs)

        label_path = os.path.join(self.path_labels, arr_labels[index])
        file = open(label_path, "r")
        file_data = file.readline()
        file.close()

        index_img = 0

        identify = label_path.replace(self.dataset_path + "labels/", "").replace(".txt", "")
        for i, image_path in enumerate(arr_images):
            if identify in image_path:
                index_img = i

        img_path = os.path.join(self.path_imgs, arr_images[index_img].replace(".txt", "jpg"))
        img = read_image(img_path)

        X = ImgTransform(img, False)
        y = file_data.split(" ") #y - labels

        cls = one_hot(torch.tensor(int(y[0])), num_classes=3)
        bbox = torch.tensor([float(x) for x in y[1:5]])
        y = torch.cat((cls, bbox))

        return X, y
    
class TestDoorsData(Dataset):
    def __init__(self, type):
        self.dataset_path = os.getcwd() + "/src/neural/dataset_test/"

        if type == "test":
            self.dataset_path += "test/"
        elif type == "valid":
            self.dataset_path += "valid/"
        else:
            self.dataset_path += "train/"

        self.path_labels = self.dataset_path + "labels/"
        self.path_imgs = self.dataset_path + "images"

        #self.img_labels, self.images = self.read_data(dataset_path + "labels/", dataset_path + "images/")

    def __len__(self):
        return len(os.listdir(self.path_labels))
    
    def __getitem__(self, index):

        arr_labels = os.listdir(self.path_labels)
        arr_images = os.listdir(self.path_imgs)

        label_path = os.path.join(self.path_labels, arr_labels[index])
        file = open(label_path, "r")
        file_data = file.readline()
        file.close()

        index_img = 0

        identify = label_path.replace(self.dataset_path + "labels/", "").replace(".txt", "")
        for i, image_path in enumerate(arr_images):
            if identify in image_path:
                index_img = i

        img_path = os.path.join(self.path_imgs, arr_images[index_img].replace(".txt", "jpg"))
        img = read_image(img_path)

        X = ImgTransform(img, False)
        y = file_data.split(" ") #y - labels

        cls = one_hot(torch.tensor(int(y[0])), num_classes=3)
        bbox = torch.tensor([float(x) for x in y[1:5]])
        y = torch.cat((cls, bbox))

        return X, y
    
def inspect_dataset(index, dataloader):
    img = dataloader.dataset[index][0]
    label = dataloader.dataset[index][1]

    img_np = ImgTransform(img, True)

    idx = torch.where(label[0:3] == 1.0000)
    if idx[0][0] == 0:
        cls = "closed"
    elif idx[0][0] == 1:
        cls = "half open"
    else: #cls == "2"
        cls = "fully open"

    x_center = round(float(label[3]) * img_np.shape[1])
    y_center = round(float(label[4]) * img_np.shape[0])
    width = round(float(label[5]) * img_np.shape[1])
    height = round(float(label[6]) * img_np.shape[0])

    start_x = x_center - round(width / 2)
    start_y = y_center - round(height / 2)

    end_x = start_x + width
    end_y = start_y + height

    cv2.rectangle(img_np, (start_x, start_y), (end_x, end_y), (0, 255, 0), 5)
    cv2.putText(img_np, cls, (start_x - 5, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("test", img_np)
    cv2.waitKey(0)

#standart case
#train_data = DoorsDataset("train")
#test_data = DoorsDataset("test")
#valid_data = DoorsDataset("valid")

#to train on better data
train_data = DoorsDataset("test")
test_data = DoorsDataset("valid")
valid_data = DoorsDataset("valid")

#train_data = TestDoorsData("test")
#test_data = TestDoorsData("valid")
#valid_data = TestDoorsData("valid")

print("train")
#train_data.__print_statistics__()
print("test")
#test_data.__print_statistics__()
print("valid")
#valid_data.__print_statistics__()

train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=False)
validation = DataLoader(valid_data, batch_size=batch_size, shuffle=False)


#for i in range(500):
#    inspect_dataset(i, test)