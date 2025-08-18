import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class LivenessNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(LivenessNet, self).__init__()
        
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes
        
        self.conv1_1 = nn.Conv2d(depth, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)
        

        self.flat_features = 32 * (height // 4) * (width // 4)
        
        self.fc1 = nn.Linear(self.flat_features, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(64, classes)
        
    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = F.softmax(self.fc2(x), dim=1)
        
        return x
    
    def load_weights(self, path):
        checkpoint = torch.load(path, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output

    @staticmethod
    def build(width, height, depth, classes):
        """
        Static method to maintain similar interface as original Keras version
        """
        return LivenessNet(width, height, depth, classes)
