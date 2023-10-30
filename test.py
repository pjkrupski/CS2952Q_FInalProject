import torch
import typer
from typing import Optional

from models import CNNModel
from preprocess import load_data

device = 'cuda' if torch.cuda.is_available() else "cpu"

def test(experiment, model, device, test_loader, loss_fn, acc_fn, verbose=False):
    model.eval()
    with experiment.test():
      test_loss = 0
      correct = 0
      test_acc = 0
      with torch.no_grad():
          for b in test_loader:
              b = b.to(device)
              out = model(b)
              test_loss += torch.sum(loss_fn(out)).item()
              correct += torch.sum(acc_fn(out)).item()

      test_loss /= len(test_loader.dataset)
      test_acc = correct / len(test_loader.dataset)
      if verbose:
          print('\nTest set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
      experiment.log_metric('Test Loss', test_loss)
      experiment.log_metric('Test Accuracy', test_acc)
      return test_loss, test_acc

def main(
    batch_size: Optional[int] = typer.Option(64, help='Input batch size for training (default: 64).'), 
    model_file: Optional[str] = typer.Option('model.pt', help='Path to model file to load for testing.')):

    _, test_loader = load_data(batch_size)

    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_file))

    loss_fn = None
    acc_fn = None

    test(model, device, test_loader, loss_fn, acc_fn, verbose=True)

if __name__ == '__main__':
    typer.run(main)