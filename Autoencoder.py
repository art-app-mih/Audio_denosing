from torch import nn


class Autoencoder(nn.Module):
    """Describe denoisng autoencoder"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(70000, 4096),
            nn.Dropout(0.3),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.Dropout(0.3))
        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.Dropout(0.3),
            nn.Linear(4096, 70000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
