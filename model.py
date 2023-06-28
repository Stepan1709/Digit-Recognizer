import torch


class LeNet33(torch.nn.Module):
    def __init__(self):
        super(LeNet33, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.bn1_1 = torch.nn.BatchNorm2d(6)
        self.act1_1 = torch.nn.Tanh()
        self.conv1_2 = torch.nn.Conv2d(
            in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(6)
        self.act1_2 = torch.nn.Tanh()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.bn2_1 = torch.nn.BatchNorm2d(16)
        self.act2_1 = torch.nn.Tanh()
        self.conv2_2 = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.bn2_2 = torch.nn.BatchNorm2d(16)
        self.act2_2 = torch.nn.Tanh()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.act3 = torch.nn.Tanh()

        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.act4 = torch.nn.Tanh()

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.act1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.act1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.act2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.act2_2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.act3(x)

        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.act4(x)

        x = self.fc3(x)

        return x
