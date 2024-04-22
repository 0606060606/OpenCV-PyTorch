import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import RandomErasing
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import torchvision
import torchsummary
data_folder_with_erasing = r"C:\Users\user\Desktop\dataset\training_dataset"
data_folder_without_erasing = r"C:\Users\user\Desktop\dataset\training_dataset"

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=None)
        self.resnet50.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.resnet50(x)


model_with_erasing = ResNet50()
model_without_erasing = ResNet50()

transform_with_erasing = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    RandomErasing(),
])

transform_without_erasing = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

dataset_with_erasing = ImageFolder(root=data_folder_with_erasing, transform=transform_with_erasing)
dataset_without_erasing = ImageFolder(root=data_folder_without_erasing, transform=transform_without_erasing)

train_loader_with_erasing = DataLoader(dataset_with_erasing, batch_size=64, shuffle=True)
train_loader_without_erasing = DataLoader(dataset_without_erasing, batch_size=64, shuffle=True)


val_dataset = ImageFolder(root=r"C:\Users\user\Desktop\dataset\validation_dataset", transform=transform_without_erasing)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


criterion = nn.CrossEntropyLoss()
optimizer_with_erasing = optim.SGD(model_with_erasing.parameters(), lr=0.001, momentum=0.9)
optimizer_without_erasing = optim.SGD(model_without_erasing.parameters(), lr=0.001, momentum=0.9)


def train_model(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


num_epochs = 30
train_model(model_with_erasing, train_loader_with_erasing, optimizer_with_erasing, criterion, num_epochs)
train_model(model_without_erasing, train_loader_without_erasing, optimizer_without_erasing, criterion, num_epochs)

val_accuracy_with_erasing = validate_model(model_with_erasing, val_loader)
val_accuracy_without_erasing = validate_model(model_without_erasing, val_loader)

torch.save(model_with_erasing.state_dict(), 'model_with_erasing.pth')
torch.save(model_without_erasing.state_dict(), 'model_without_erasing.pth')


labels = ['Random Erasing', 'Without Random Erasing']
accuracies = [val_accuracy_with_erasing, val_accuracy_without_erasing]

plt.bar(labels, accuracies, color=['blue', 'blue'])
plt.ylabel('Accuracy')
plt.title('Accuracy comparison ')
plt.show()