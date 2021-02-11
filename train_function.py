import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()  # Enter train mode
        loss_accum = 0
        correct_samples = 0
        total_samples = 0

        for i_step, (x, y, _) in enumerate(train_loader):

            x_gpu = x.to(device).float()
            y_gpu = y.to(device).float()

            prediction = model(x_gpu).float()
            loss_value = loss(prediction, y_gpu).float()

            optimizer.zero_grad()
            loss_value.backward()
            train_loss += loss_value.item()
            optimizer.step()

            val_loss += compute_loss(model, val_loader, loss)

            if epoch % 2 == 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_value,
                            }, f'model_test_{epoch}.pth')

            scheduler.step()

        if epoch % 2 == 0:
            print('====> Epoch: {} Average train loss: {:.4f} , Average val loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset), val_loss / len(val_loader.dataset)))


def compute_loss(model, loader, loss):
    """
    Computes loss on the dataset wrapped in a loader
    """
    model.eval()  # Evaluation mode

    for i_step, (x, y, _) in enumerate(loader):
        x_gpu = x.to(device).float()
        y_gpu = y.to(device).float()
        pred_val = model(x_gpu).float()
        loss_value = loss(pred_val, y_gpu)

    return loss_value
