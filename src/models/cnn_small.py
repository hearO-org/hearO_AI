import torch, torch.nn as nn, torch.nn.functional as F

class CNN_Small(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, num_filters=(32,64,128), dropout=0.25):
        super().__init__()
        C1, C2, C3 = num_filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, C1, 3, padding=1), nn.BatchNorm2d(C1), nn.ReLU(), nn.MaxPool2d((2,2)))
        self.block2 = nn.Sequential(
            nn.Conv2d(C1, C2, 3, padding=1), nn.BatchNorm2d(C2), nn.ReLU(), nn.MaxPool2d((2,2)))
        self.block3 = nn.Sequential(
            nn.Conv2d(C2, C3, 3, padding=1), nn.BatchNorm2d(C3), nn.ReLU(), nn.AdaptiveMaxPool2d((1, None)))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(C3, num_classes)

    def forward(self, x):            # x: (B,1,M,T)
        x = self.block1(x)           # (B,C1,M/2,T/2)
        x = self.block2(x)           # (B,C2,M/4,T/4)
        x = self.block3(x)           # (B,C3,1,T')
        x = x.squeeze(2)             # (B,C3,T')
        x = x.mean(-1)               # (B,C3)  (T' 평균 풀링)
        x = self.dropout(x)
        return self.classifier(x)
