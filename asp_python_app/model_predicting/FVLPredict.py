from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch
from pathlib import Path
from torchvision.transforms import v2
import cv2
from PIL import Image
import numpy as np
import glob

parent_directory = Path(__file__).resolve().parent

class FVLPredictDataset(Dataset):
    '''
    Loads images from folder and normalizes data, similar to ImageDataGenerator
    '''
    def __init__(self, image_size=(256, 256), image_path=".", image_suffix="_flow_loop.png"):
        self.images = []
        self.image_names = []
        print(str(image_path) + '*' + image_suffix)
        for fvl_img_file in glob.glob(str(image_path) + '/*' + image_suffix):
            print("Adding: " + fvl_img_file)
            image = Image.open(fvl_img_file)
            image = np.array(image)
            if len(image.shape) == 2:
                copied_images = [image.copy() for _ in range(3)]
                image = np.stack(copied_images, axis=-1)
            image = cv2.resize(image[:,:,:3], image_size)
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image)
            self.images.append(image)
            self.image_names.append(Path(fvl_img_file).name.removesuffix(image_suffix))
        self.images = np.array(self.images)
        self.normalize()

    def normalize(self):
        self.images = self.images/255.0

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return {"images": self.images[index], "names": self.image_names[index]}
    
    def __add__(self, other):
        self.images = np.concatenate((self.images, other.images), axis=0)
        return self

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

def predict_with_model(model, data_loader, device):
    model.eval()
    predictions_labels = []
    predictions_probs = []
    file_names = []

    with torch.no_grad():
        for D in data_loader:
            images = D["images"].to(device)
            names = D["names"]
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()

            predictions_labels.extend(predicted_labels.cpu().numpy())
            predictions_probs.extend(predicted_probs.cpu().numpy())
            file_names.extend(names)

    predictions_labels = np.array(predictions_labels)
    predictions_probs = np.array(predictions_probs)
    # print(predictions_labels)
    # print(predictions_probs)
    # print(file_names)
    return predictions_labels, predictions_probs, file_names

def predict_from_directory(dir_path, model_path=parent_directory/"model/FVL_classifier_97.pth.tar"):
    dataset = FVLPredictDataset(image_path=dir_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = MLC(num_classes=2)
    model = model.to(device)
    params = torch.load(model_path)
    model.load_state_dict(params['state_dict'])
    return predict_with_model(model, dataloader, device)

if __name__ == "__main__":
    predict_from_directory(parent_directory/"test_images")