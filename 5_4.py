import torch
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
import random
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
from typing import OrderedDict
# import torch.utils.data as data
# import torchvision.transforms as transforms
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QFileDialog

import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torch.nn as nn
from torchsummary import summary
from PyQt5.QtWidgets import QFileDialog
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QDialog
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn

class Ui_Dialog(object):
    def __init__(self):
        # Define transform here
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.verticalLayout = QVBoxLayout(Dialog)

        # Create QLabel for displaying the loaded image
        self.image_label = QLabel(Dialog)
        self.verticalLayout.addWidget(self.image_label)

        # Button to load an image
        self.load_image_button = QPushButton("Load Image", Dialog)
        self.load_image_button.clicked.connect(self.load_image)
        self.verticalLayout.addWidget(self.load_image_button)
        self.prediction_label = QLabel(Dialog)
        self.verticalLayout.addWidget(self.prediction_label)

        # Button to predict
        self.predict_button = QPushButton("Predict", Dialog)
        self.predict_button.clicked.connect(self.predict)
        self.verticalLayout.addWidget(self.predict_button)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ResNet50 Prediction"))

    def load_image(self):
        # Open a file dialog to select an image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(None, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        
        # Display the loaded image
        pixmap = QtGui.QPixmap(file_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

        # Store the file_path for later use in prediction
        self.image_path = file_path

    def predict(self):
        # Check if an image is loaded
        if not hasattr(self, 'image_path'):
            print("Please load an image first.")
            return

        # Perform prediction on the loaded image
        input_image = Image.open(self.image_path).convert('RGB')
        input_tensor = self.transform(input_image).unsqueeze(0)

        with torch.no_grad():
            output_with_erasing = model_with_erasing(input_tensor)
            output_without_erasing = model_without_erasing(input_tensor)

        predicted_class_without_erasing_name = self.get_predicted_class_name(output_without_erasing)
        self.prediction_label.setText(f"Predict (Without Random Erasing): {predicted_class_without_erasing_name}")

        print("Predict (Without Random Erasing):", predicted_class_without_erasing_name)

    def get_predicted_class_name(self, output_tensor):
        probabilities = torch.nn.functional.softmax(output_tensor[0], dim=0)
        predicted_class_threshold = 1 if probabilities[1] > threshold else 0
        predicted_class_name = class_mapping.get(predicted_class_threshold, "Unknown")
        return predicted_class_name
        






class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=None)
        self.resnet50.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.resnet50(x)



model_with_erasing = ResNet50()
model_without_erasing = ResNet50()


model_without_erasing.load_state_dict(torch.load(r"C:\Users\user\Desktop\opencv HW2  66\Q5_model_without_erasingbest.pth"))


model_with_erasing.eval()
model_without_erasing.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


image_path = r"C:\Users\user\Desktop\dataset\inference_dataset\Cat\8043.jpg"
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)


with torch.no_grad():
    output_with_erasing = model_with_erasing(input_tensor)
    output_without_erasing = model_without_erasing(input_tensor)

predicted_class_with_erasing = torch.argmax(output_with_erasing).item()
predicted_class_without_erasing = torch.argmax(output_without_erasing).item()

threshold = 0.9  

probabilities_with_erasing = torch.nn.functional.softmax(output_with_erasing[0], dim=0)
probabilities_without_erasing = torch.nn.functional.softmax(output_without_erasing[0], dim=0)


class_mapping = {0: "Cat", 1: "Dog"}

predicted_class_with_erasing_threshold = 1 if probabilities_with_erasing[1] > threshold else 0  # Assuming binary classification
predicted_class_without_erasing_threshold = 1 if probabilities_without_erasing[1] > threshold else 0  # Assuming binary classification


predicted_class_with_erasing_name = class_mapping.get(predicted_class_with_erasing_threshold, "Unknown")
predicted_class_without_erasing_name = class_mapping.get(predicted_class_without_erasing_threshold, "Unknown")

print("Predict (Without Random Erasing):", predicted_class_without_erasing_name)
def main():
    app = QApplication(sys.argv)
    dialog = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(dialog)
    dialog.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()