import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, N_CLASS):
        super(CNN, self).__init__()

        ## Convolutuin layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        

        ## Maxpooling
        self.Maxpool = nn.MaxPool2d(2, 2)
        #self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((7, 7))

        ## Activation Function
        self.ReLu = nn.ReLU(inplace=False)
        #self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((7, 7))

        ## Dropout
        self.dropout = nn.Dropout(0.5)

        ## Fully connected layers:
        self.fc1 = nn.Linear(10*117*64, 64)
        self.fc2 = nn.Linear(64, N_CLASS)
    
    def forward(self, x, training=True):
        
        x = self.ReLu(self.conv1(x))
        x = self.Maxpool(x)
        x = self.ReLu(self.conv2(x))
        x = self.Maxpool(x)
        x = self.ReLu(self.conv3(x))
        #x = self.AdaptiveAvgPool(x)
        
        #x = x.view(-1, 5*117*64)
        x = x.view(-1, 10*117*64)
        x = self.ReLu(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLu(self.fc2(x))
        x = F.softmax(x, dim=1)
  
        return x

    