from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from hw2ui import Ui_MainWindow
import cv2
import numpy as np
import math
import os
import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import torchsummary
import matplotlib.pyplot as plt
import io
from torchvision.models import resnet50
from torchvision.transforms import transforms
from PIL import Image
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(weights=None)
        self.resnet50.fc = nn.Linear(2048, 2)

    def forward(self, x):
        return self.resnet50(x)
class DigitClassifier:
    def __init__(self, model_path=r"C:\Users\user\Desktop\opencv HW2  66\Q4_best_model_weights.pth", num_classes=10):
        
        
        self.model = SimpleCNN(num_classes=num_classes).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        
        input_image = Image.open(image_path).convert("L")
        input_tensor = self.transform(input_image).unsqueeze(0).to(device)

        
        with torch.no_grad():
            model_output = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(model_output, dim=1)[0].cpu().numpy()

        
        plt.bar(range(len(probabilities)), probabilities, tick_label=[str(i) for i in range(len(probabilities))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Predicted Probability Distribution')
        plt.show()





      
        _, predicted_class = model_output.max(1)
        predicted_class = predicted_class.item()

        return predicted_class




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setup_control()
        
        
    def setup_control(self):
        # TODO
        #self.ui.textEdit.setText('Happy World!')
        self.ui.load_image_1.clicked.connect(self.open_file1)
        self.ui.load_image_2.clicked.connect(self.open_file2)
        self.ui.draw_contour.clicked.connect(self.draw_contour)
        self.ui.count_coin.clicked.connect(self.count_coin)

        self.ui.show_model.clicked.connect(self.show_model)#4.1
        self.ui.Show_Accuracy.clicked.connect(self.Show_Accuracy)#4.2
        self.ui.predict.clicked.connect(self.predict)#4.3
        self.ui.reset.clicked.connect(self.reset)#4.4

        self.ui.stereo_disparity_map_2.clicked.connect(self.show_stereo)#22222
        self.ui.reset.clicked.connect(self.reset)
        self.ui.closing.clicked.connect(self.closing)
        self.ui.Opening.clicked.connect(self.opening)

        #55555555555555555555555555555555555555555555555 Show_Comparison
        self.ui.show_image51.clicked.connect(self.show_image51)
        self.ui.Show_model_structure.clicked.connect(self.Show_model_structure)
        self.ui.Show_Comparison.clicked.connect(self.Show_Comparison)
        
       

    
    
    
    

    def open_file1(self):
        global filename1
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")  
        print(filename, filetype)
        filename1 = filename
        self.ui.image_label_1.setText(filename)

    def open_file2(self):#555555555555555555555555555555555555555555555555555
        global filename2
        filename, filetype = QFileDialog.getOpenFileName(self,"Open file","./")  
        print(filename, filetype)
        filename2 = filename
        self.ui.image_label_2.setText(filename)

    def load_model_weights(self):

        self.model = SimpleCNN(num_classes=10).to(self.device)

        
        weights_path = "simple_cnn_mnist_weights.pth"
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    #1111111111111111111111111111111111111111111111111111111111111111111111111111111

    def draw_contour(self):#1.1--------------------------------------------------------------------------
        import cv2
        import numpy as np

        
        image = cv2.imread(filename1)
        resized_original = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original Image', resized_original)

        # 轉灰接
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #Canny檢測邊緣
        edges = cv2.Canny(blurred, 50, 150)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=10, maxRadius=50)
        circle_center_image = np.zeros_like(image)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 畫圓外圍
                cv2.circle(circle_center_image, (i[0], i[1]), 2, (0, 0, 255), 3) # 畫圓心

        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        resized_circle_center = cv2.resize(circle_center_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        
        cv2.imshow('Processed Image', image)
        cv2.imshow('Circle Center Image', resized_circle_center)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

    def count_coin(self):#1.2-----------------------------------------------------------------------
        import cv2
        import numpy as np
        img = cv2.imread(filename1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=10, maxRadius=50)
        
        
        if circles is not None:
            num_coins = len(circles[0])
        else:
            num_coins = 0
        self.ui.count_coin_label_1.setText("There are " + str(num_coins) + " coins in image")
        
        
    #2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222


    def show_stereo(self):#2222222222222222222222222222222222222222222222222222222222
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        
        original_image = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)

        
        equalized_image_opencv = cv2.equalizeHist(original_image)

        hist, bins = np.histogram(original_image.flatten(), 256, [0, 256])
        pdf = hist / np.sum(hist)
        cdf = np.cumsum(pdf)
        lookup_table = np.uint8(255 * cdf)
        equalized_image_manual = cv2.LUT(original_image, lookup_table)

        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

        plt.subplot(231), plt.imshow(original_image, cmap='gray'), plt.title('Original Image')
        plt.subplot(232), plt.imshow(equalized_image_opencv, cmap='gray'), plt.title('Equalized Image (OpenCV)')
        plt.subplot(233), plt.imshow(equalized_image_manual, cmap='gray'), plt.title('Equalized Image (Manual)')

        plt.subplot(234), plt.hist(original_image.flatten(), 255, [0, 256], color='b', alpha=0.5, label='Original Histogram')
        plt.subplot(235), plt.hist(equalized_image_opencv.flatten(), 255, [0, 256], color='b', alpha=0.5, label='Equalized Histogram (OpenCV)')
        plt.subplot(236), plt.hist(equalized_image_manual.flatten(), 255, [0, 256], color='b', alpha=0.5, label='Equalized Histogram (Manual)')

        plt.subplot(234), plt.hist(original_image.flatten(), 255, [0, 256], color='b', alpha=0.5, )
        plt.xlabel('Grayscale Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        plt.subplot(235), plt.hist(equalized_image_opencv.flatten(), 255, [0, 256], color='b', alpha=0.5, )
        plt.xlabel('Grayscale Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        plt.subplot(236), plt.hist(equalized_image_manual.flatten(), 255, [0, 256], color='b', alpha=0.5, )
        plt.xlabel('Grayscale Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        
        plt.show()


    #44444444444444444444444444444444444444444444444444444444444444444444444444444444
    def show_model(self):#4.1
        device=torch.device("cuda" if torch.cuda.is_available () else"cpu")


        model = torchvision.models.vgg19_bn(num_classes=10)
        inputs=model.to(device)

        
        torchsummary.summary(model, (3, 32, 32))

    def Show_Accuracy(self):#4.2----------------------------------------------------------------------------
        import cv2

        
        image_path = r"C:\Users\user\Desktop\opencv HW2  66\4_2 loss and accuracy.png"

        
        image = cv2.imread(image_path)

        
            
        cv2.imshow('4_2', image)
            
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def predict(self):#4.3-------------------------------------------------------------------------
        
        
        pixmap = self.ui.drawing_board_widget.drawing_board.get_pixmap()
       
        image = pixmap.toImage()

       
        image_path = r"C:\Users\user\Desktop\predicted_image.png"
        image.save(image_path, "PNG")

        classifier = DigitClassifier()
        predicted_class = classifier.predict(image_path)

        print(f"The predicted class for the input image is: {predicted_class}")
        self.ui.prediction_result_label.setText(f"Predicted digit: {predicted_class}")

    
                
    
    def reset(self):#4.4--------------------------------------------------------------------------------
        self.ui.drawing_board_widget.drawing_board.reset_board()
    
    #33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

    def closing(self):#3.1----------------------------------------------------------------------------------
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt

        
        image = cv2.imread(filename1)

        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        K = 3
        padded_image = np.pad(binary_image, (K // 2, K // 2), mode='constant', constant_values=0)

        
        kernel = np.ones((K, K), np.uint8)

        dilated_image = np.zeros_like(padded_image)
        for i in range(padded_image.shape[0] - K + 1):
            for j in range(padded_image.shape[1] - K + 1):
                if np.any(padded_image[i:i+K, j:j+K] * kernel):
                    dilated_image[i, j] = 255

        plt.subplot(121), plt.imshow(binary_image, cmap='gray'), plt.title('Original Image')
        plt.subplot(122), plt.imshow(dilated_image, cmap='gray'), plt.title('result')
        plt.show()
    
    def opening(self):#3.2---------------------------------------------------------------------
        import numpy as np
        import cv2
        from matplotlib import pyplot as plt

        
        original_image = cv2.imread(filename1)
        cv2.imshow('Original Image', original_image)

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        kernel_size = 3
        padded_image = np.pad(binary_image, (kernel_size // 2, kernel_size // 2), mode='constant')

        structuring_element = np.ones((3, 3), np.uint8)

        erosion_result = np.zeros_like(padded_image)
        for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
                if np.all(padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1, j - kernel_size // 2:j + kernel_size // 2 + 1]):
                    erosion_result[i, j] = 255

        plt.imshow(erosion_result, cmap='gray'), plt.title('Result')

        plt.show()

    def show_image51(self):#5.1-------------------------
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
        
        dataset = torchvision.datasets.ImageFolder(r"C:\Users\user\Desktop\dataset\inference_dataset",transform=transform) 
        Inferenceloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,num_workers=2)
        
        classes = ('cat', 'dog')
        
        for i, (batch_x, batch_y) in enumerate(Inferenceloader):
            if batch_y == 0:
                plt.subplot(1,2,1)
                grid = torchvision.utils.make_grid(batch_x,nrow=2)
                plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.title(classes[batch_y])
                plt.axis('off')
            else :
                plt.subplot(1,2,2)
                grid = torchvision.utils.make_grid(batch_x,nrow=2)
                plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.title(classes[batch_y])
                plt.axis('off')
        plt.show()

    def Show_model_structure(self):#5.2---------------------------------------------
        import torch.nn as nn
        model = torchvision.models.resnet50() 
        model.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        summary(model,input_size=(3, 224,224))

    def Show_Comparison(self):#5.3-----------------------------------------------------
        import cv2
        image_path = r"C:\Users\user\Desktop\opencv HW2  66\5_3 accuracy.png"
        image = cv2.imread(image_path)
        cv2.imshow('5_3', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())