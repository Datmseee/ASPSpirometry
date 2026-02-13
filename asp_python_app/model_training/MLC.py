import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision import models
from torchvision.models import resnet
from torchvision.models import ResNet18_Weights

class MLC(nn.Module):
    def __init__(self, num_classes):
        super(MLC, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def test_model(model, data_loader, device, class_labels):
    model.eval()
    test_predictions = []
    test_labels = []
    class_accuracy = []
    class_f1_score = []
    test_predictions_probs = []

    with torch.no_grad():
        for D in data_loader:
            images = D["images"].to(device)
            labels = D["labels"].to(device)
            outputs = model(images)
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()

            test_predictions.extend(predicted_labels.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_predictions_probs.extend(predicted_probs.cpu().numpy())
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)
    test_predictions_probs = np.array(test_predictions_probs)
    macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Total Accuracy: {accuracy:.4f}")

    # Calculate prediction accuracy, precision, and F1 score for each class
    for i, class_label in enumerate(class_labels):
        class_accuracy.append(accuracy_score(test_labels[:, i], test_predictions[:, i]))
        class_f1_score.append(f1_score(test_labels[:, i], test_predictions[:, i]))

    print('Model Macro F1-score: %.4f' % macro_f1)
    print('Prediction Metrics per Class:')
    for i, class_label in enumerate(class_labels):
        print('%s - Accuracy: %.4f, F1-score: %.4f' % (
            class_label, class_accuracy[i], class_f1_score[i]))

    # Compute the classification report
    class_report = classification_report(test_labels, test_predictions, target_names=class_labels, digits=3)

    print('Classification Report:')
    print(class_report)

    return test_labels, test_predictions, class_accuracy, class_f1_score