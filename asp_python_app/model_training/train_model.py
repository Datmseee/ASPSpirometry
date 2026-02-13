from FVLModelTrainingDataset import FVLModelTrainingDataset
from MLC import MLC, EarlyStopper, save_checkpoint, test_model
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import ConcatDataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

augment = v2.Compose([
    v2.ColorJitter(0, 0.25, 0.25, 0.5),
    v2.RandomChannelPermutation(),
    v2.RandomInvert(),
    v2.RandomGrayscale(),
])

image_size = (256, 256)

dataset = FVLModelTrainingDataset(image_size, transform=augment)
dataset.train_val_test_split()
class_labels = dataset.classes
for _ in range(10):
    temp_dataset = FVLModelTrainingDataset(image_size, transform=augment)
    temp_dataset.train_val_test_split()
    dataset += temp_dataset
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# print(f"Current device number: {torch.cuda.current_device()}")
# print(f"Device name: {torch.cuda.get_device_name()}")

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

num_classes = 2
model = MLC(num_classes=num_classes)
model = model.to(device)

eta = 0.000005
num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=eta)

epoch_train_loss = []
epoch_val_loss = []
early_stopper = EarlyStopper(patience=5)
lowest_val_loss_checkpoint = 0
# dataset.mode = "train"
# print(len(dataset))
# dataset.mode = "val"
# print(len(dataset))
# dataset.mode = "test"
# print(len(dataset))
# print(dataset.labels[100:150,:])
error = nn.BCEWithLogitsLoss()
for epoch in range(num_epochs):
    dataset.mode = 'train'
    train_losses = []
    model.train()
    for D in dataloader:
        optimizer.zero_grad()
        data = D['images'].to(device, dtype=torch.float)
        labels = D['labels'].to(device, dtype=torch.float)
        y_hat = model(data)
        loss = torch.sum(error(y_hat, labels))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    mean_train_losses = np.mean(train_losses)
    epoch_train_loss.append(mean_train_losses)
    val_losses = []
    dataset.mode = 'val'
    model.eval()
    with torch.no_grad():
        for D in dataloader:
            data = D['images'].to(device, dtype=torch.float)
            label = D['labels'].to(device, dtype=torch.float)
            y_hat = model(data)
            loss = torch.sum(error(y_hat, labels))
            val_accuracy = accuracy_score(label.cpu().numpy(), (y_hat > 0.5).float().cpu().numpy())
            val_losses.append(loss.item())
    mean_val_losses = np.mean(val_losses)
    
    epoch_val_loss.append(mean_val_losses)
    print(f'Train Epoch: {epoch+1}\tTrain Loss: {mean_train_losses:.6f}\tVal Acc: {val_accuracy:.6f}\tVal Loss: {mean_val_losses:.6f}')
    if early_stopper.early_stop(mean_val_losses):
        break
    checkpoint_filename = f"./checkpoints/checkpoint_epoch_{epoch+1}_{mean_train_losses:.4f}_{mean_val_losses:.4f}.pth.tar"
    checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    save_checkpoint(checkpoint, filename=checkpoint_filename)
    if early_stopper.min_validation_loss == mean_val_losses:
        lowest_val_loss_checkpoint = checkpoint_filename

best_checkpoint = torch.load(checkpoint_filename)
model.load_state_dict(best_checkpoint['state_dict'])
optimizer.load_state_dict(best_checkpoint['optimizer'])
model.eval()
dataset.mode = "test"

test_labels, test_predictions, class_accuracy, class_f1_score = test_model(model, dataloader, device, class_labels)