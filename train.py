import torch
from tqdm import tqdm
import typer
from typing import Optional

from models import CNNModel
from preprocess import load_data
from test import test

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Train/test code courtesy of pytorch examples repo. (https://github.com/pytorch/examples/blob/main/mnist/main.py#L12)
def train(args, model, device, train_loader, optimizer, loss_fn):
    """
    :param args command line arguments
    :param model model to be trained
    :param device either cuda or CPU depending on availability
    :param train_loader a Pytorch Dataloader of the trainset and labels
    :param optimizer nn.optimizer (Adam)
    :param loss_fn Loss function to train model to.
    :param writer The tensorboard SummaryWriter to log data on.
    """
    model.train()

    for _ in range(args['epochs']):
        for b in tqdm(train_loader):
            b = b.to(device)
            optimizer.zero_grad()

            out = model(b)
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()

def main(
    batch_size: Optional[int] = typer.Option(16, help='Input batch size for training (default: 64).'), 
    epochs: Optional[int] = typer.Option(10, help='Number of epochs to train (default: 10).'), 
    lr: Optional[float] = typer.Option(2e-5, help='Learning rate (default: 0.1).'), 
    seed: Optional[int] = typer.Option(1, help='Random seed (default: 1).'),
    log_interval: Optional[int] = typer.Option(10, help='how many batches to wait before logging training status (default: 10).'),
    save_model: Optional[bool] = typer.Option(True, help='For saving the current model.'),
    output_file: Optional[str] = typer.Option('model.pt', help='The name of output file.')):

    args = {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'seed': seed,
        'log_interval': log_interval,
        'save_model': save_model,
    }
    torch.manual_seed(seed)

    train_loader, test_loader = load_data(batch_size)

    model = CNNModel().to(device)
   
    loss_fn = None
    acc_fn = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train(args, model, device, train_loader, optimizer, loss_fn)

    if save_model:
        torch.save(model.state_dict(), output_file)

    test(model, device, test_loader, loss_fn, acc_fn, verbose=True)

if __name__ == '__main__':
    typer.run(main)
