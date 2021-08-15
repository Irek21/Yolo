import torch
import torch.nn as nn
import torchvision

class Detector(nn.Module):
    def __init__(self):
        
        cnn = torchvision.models.vgg16(pretrained=True)
        conv = nn.Sequential(
            cnn.features,
            cnn.avgpool,
            nn.Flatten(),
        )

        fc = nn.Sequential(
            nn.Linear(49 * 512, 4096),
            nn.LeakyReLU(0.1),

            nn.Linear(4096, 7 * 7 * 30),
        )
        
        super(Detector, self).__init__()
        self.conv = conv
        self.fc = fc
        
    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out).reshape(-1, 7, 7, 30)
        return out