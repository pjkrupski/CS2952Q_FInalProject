import torch
import typer
from typing import Optional
import torch.nn as nn
from torchmetrics.classification import MulticlassConfusionMatrix

from models import CNNModel_128, VitModel
from preprocess import load_single_data

device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else "cpu"

def test(model, device, test_loader, loss_fn, acc_fn):
    conf_mat = MulticlassConfusionMatrix(num_classes=10)
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
            conf_mat.update(out, target)

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    fig_, ax_ = conf_mat.plot()
    return test_loss, test_acc

def main(
    batch_size: Optional[int] = typer.Option(64, help='Input batch size for training (default: 64).'), 
    model_name: Optional[str] = typer.Option('cnn', help="cnn or vit"),
    model_file: Optional[str] = typer.Option('128b_120e.pt', help='Path to model file to load for testing.')):

    torch.manual_seed(1)
    _, test_loader = load_single_data(batch_size)

    model = None
    if model_name.lower() == "cnn":
        model = CNNModel_128(False, 1).to(device)
    else:
        model = VitModel.to(device)
    model.load_state_dict(torch.load(model_file))

    loss_fn = nn.CrossEntropyLoss()
    acc_fn = lambda out, target: nn.functional.one_hot(torch.argmax(out, dim=1), 10) * target

    test(model, device, test_loader, loss_fn, acc_fn)

if __name__ == '__main__':
    typer.run(main)