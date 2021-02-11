import torch
from torch import nn
from Audio_dataset import AudioDataset
from Autoencoder import Autoencoder

test_data = AudioDataset('/content/drive/MyDrive/Colab Notebooks/Denoising/test', SNR=8)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(torch.load('model_test_{epoch}.pth'))
for param in model.features.parameters():
    param.requires_grad = False

loss = nn.MSELoss()
loss_value = 0

for i_step, (x, y, _) in enumerate(test_loader):
  model.eval()
  model.cpu()

  prediction = model(x.float()).float()
  loss_value += loss(prediction, y).float()

print('Mean loss on test data:', loss_value / len(test_loader.dataset))
