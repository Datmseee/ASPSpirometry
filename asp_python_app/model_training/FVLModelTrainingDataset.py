from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
import cv2
from PIL import Image
import numpy as np
import json

class FVLModelTrainingDataset(Dataset):
    def __init__(self, image_size, transform=None):
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = None, None, None, None, None, None
        self.mode = "all"
        self.transform = transform
        self.classes, self.labels, self.images = [], [], []

        images_path = "./dataset/images"
        labels_path = "./dataset/labels.json"
    
        with open(labels_path) as f:
            labels_json = json.load(f)

        mlb = MultiLabelBinarizer()
        self.labels = np.array(mlb.fit_transform(labels_json.values()))
        self.classes = mlb.classes_

        for img_name in labels_json.keys():
            image_path = os.path.join(images_path, img_name)
            image = Image.open(image_path)
            image = np.array(image)
            if len(image.shape) == 2:
                copied_images = [image.copy() for _ in range(3)]
                image = np.stack(copied_images, axis=-1)
            image = cv2.resize(image[:,:,:3], image_size)
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)
            # cv2.imwrite(os.path.join("./output", img_name), image)
            if self.transform:
                image = transform(image)
                # save_image(image, os.path.join('./temp',str(img_name)))
            self.images.append(image)
        self.images = np.array(self.images)
        self.normalize()

    def normalize(self):
        self.images = self.images/255.0
    
    def train_val_test_split(self):
        # 0.1 0.1 0.8 split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size=0.1)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.images, self.labels, test_size=0.11)

    def __len__(self):
        match self.mode:
            case "train":
                return self.x_train.shape[0]
            case "val":
                return self.x_val.shape[0]
            case "test":
                return self.x_test.shape[0]
        return self.images.shape[0]
    
    def __getitem__(self, index):
        match self.mode:
            case "train":
                return {"images": self.x_train[index], "labels": self.y_train[index]}
            case "val":
                return {"images": self.x_val[index], "labels": self.y_val[index]}
            case "test":
                return {"images": self.x_test[index], "labels": self.y_test[index]}
        return {"images": self.images[index], "labels": self.labels[index]}
    
    def __add__(self, other):
        self.x_train = np.concatenate((self.x_train, other.x_train), axis=0)
        self.x_val = np.concatenate((self.x_val, other.x_val), axis=0)
        self.x_test = np.concatenate((self.x_test, other.x_test), axis=0)
        self.y_train = np.concatenate((self.y_train, other.y_train), axis=0)
        self.y_val = np.concatenate((self.y_val, other.y_val), axis=0)
        self.y_test = np.concatenate((self.y_test, other.y_test), axis=0)
        self.images = np.concatenate((self.images, other.images), axis=0)
        self.labels = np.concatenate((self.labels, other.labels), axis=0)
        return self