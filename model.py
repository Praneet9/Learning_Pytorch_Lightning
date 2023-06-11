import torch.nn as nn


class Model(nn.Module):
    def __init__(self, classes=12):
        super(Model, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((3, 3)),
            nn.Dropout2d(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((3, 3)),
            nn.Dropout2d(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 5, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.classifier = nn.Linear(64, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.fc_block(x)
        output = self.softmax(self.classifier(x))
        return output


if __name__ == "__main__":
    import torch

    model = Model().cuda()
    tensor = torch.rand((1, 1, 98, 50)).cuda()
    model(tensor)
