import torch
import typer
from typing import Optional
import torch.nn as nn

from models import CNNModel_128, CNNModel_640
from preprocess import load_single_data, load_multi_data

device = 'cuda' if torch.cuda.is_available() else "cpu"

def test(model, device, test_loader, loss_fn, acc_fn):
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = 0
    with torch.no_grad():
        for data, target, filename in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += torch.sum(loss_fn(out, target)).item()
            correct += torch.sum(acc_fn(out, target)).item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_acc

def main(
    batch_size: Optional[int] = typer.Option(64, help='Input batch size for training (default: 64).'), 
    model_file: Optional[str] = typer.Option('model.pt', help='Path to model file to load for testing.')):

    _, test_loader = load_single_data(batch_size)

    model = CNNModel_128().to(device)
    model.load_state_dict(torch.load(model_file))

    loss_fn = nn.CrossEntropyLoss()
    acc_fn = lambda out, target: nn.functional.one_hot(torch.argmax(out, dim=1), 10) * target

    test(model, device, test_loader, loss_fn, acc_fn)

if __name__ == '__main__':
    typer.run(main)