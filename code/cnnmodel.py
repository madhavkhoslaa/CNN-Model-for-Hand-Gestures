import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)

        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)

        self.cnn4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=3)

        self.fn_1 = nn.Linear(32, 28)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)

        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        out = out.view(out.size(0), -1)
        out = self.fn_1(out)
        return out
