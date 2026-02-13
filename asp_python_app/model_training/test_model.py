from FVLModelTrainingDataset import FVLModelTrainingDataset
from MLC import MLC, EarlyStopper, save_checkpoint, test_model
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import ConcatDataset
import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader

augment = v2.Compose([
    v2.ColorJitter(0, 0.25, 0.25, 0.5),
    v2.RandomChannelPermutation(),
    v2.RandomInvert(),
    v2.RandomGrayscale(),
])

image_size = (256, 256)

dataset = FVLModelTrainingDataset(image_size)
dataset.train_val_test_split()
labels = dataset.classes
# for _ in range(10):
#     temp_dataset = FVLDataset(image_size, transform=augment)
#     temp_dataset.train_val_test_split()
#     dataset += temp_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# print(f"Current device number: {torch.cuda.current_device()}")
# print(f"Device name: {torch.cuda.get_device_name()}")

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_classes = 2
model = MLC(num_classes=num_classes)
model = model.to(device)

eta = 0.000005
optimizer = torch.optim.Adam(model.parameters(), lr=eta)

checkpoint = torch.load('./checkpoints/checkpoint_epoch_5_0.1400_1.4270.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
dataset.mode = "all"

test_labels, test_predictions, class_accuracy, class_f1_score = test_model(model, dataloader, device, labels)
print(multilabel_confusion_matrix(test_labels, test_predictions))