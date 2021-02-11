import torch
from torch import nn
from Audio_dataset import AudioDataset
from Autoencoder import Autoencoder
from train_function import train_model

train_data = AudioDataset('/path/train', SNR=8)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=12)

val_data = AudioDataset('/path/val', SNR=8)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
loss = nn.MSELoss()

# Configure the optimiser
learning_rate = 1e-3

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epochs=30)
