from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn as nn
import torch
from pathlib import Path
from torchvision.transforms import v2
import cv2
from PIL import Image
import numpy as np
import glob

parent_directory = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = parent_directory / "model" / "FVL_classifier_99_0.04.pth.tar"

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
        if self.images:
            self.images = torch.stack(self.images, dim=0)
            # Keep preprocessing consistent with the original training pipeline.
            self.images = self.images / 255.0
        else:
            self.images = torch.empty((0, 3, image_size[1], image_size[0]), dtype=torch.float32)

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return {"images": self.images[index], "names": self.image_names[index]}
    
    def __add__(self, other):
        self.images = torch.cat((self.images, other.images), dim=0)
        return self

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=None)
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

def _resolve_model_path(model_path=None):
    if model_path is not None:
        return Path(model_path)

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    raise FileNotFoundError(f"Model checkpoint not found: {DEFAULT_MODEL_PATH}")


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dictionary.")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint

    raise KeyError("Checkpoint does not contain a supported state dict.")


def _normalize_state_dict(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    return state_dict


def predict_from_directory(dir_path, model_path=None):
    dataset = FVLPredictDataset(image_path=dir_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = MLC(num_classes=2)
    model = model.to(device)
    model_path = _resolve_model_path(model_path)
    params = torch.load(model_path, map_location=device)
    state_dict = _normalize_state_dict(_extract_state_dict(params))
    model.load_state_dict(state_dict)
    return predict_with_model(model, dataloader, device)

if __name__ == "__main__":
    predict_from_directory(parent_directory/"test_images")
