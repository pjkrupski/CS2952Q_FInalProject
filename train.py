import torch
import random
from tqdm import tqdm
import typer
from typing import Optional
import torch.nn as nn
import numpy as np

from models import CNNModel_128, CNNModel_640, VitModel
from preprocess import load_single_data, load_multi_data
from test import test


from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

device = 'cuda' if torch.cuda.is_available() else "cpu"

# Train/test code courtesy of pytorch examples repo. (https://github.com/pytorch/examples/blob/main/mnist/main.py#L12)
def train(args, model, device, train_loader, test_loader, optimizer, loss_fn, acc_fn):
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

    #train loader len is 767
    #epochs is 10
    #7582 images in train folder 
       
    num_images = len(train_loader) * args['batch_size'] 
    num_images_toperturb = num_images * (args['pjgd_training']/100)
    batches_perturbed_perEpoch = int(num_images_toperturb / args['batch_size'])
    total_batches_perturbed = 0

    for e in range(args['epochs']):
        print(f"Epoch {e}")
        train_loss = 0
        correct = 0
        batches_to_perturb = set()
        batches_to_perturb.update(random.sample(range(0, len(train_loader)), batches_perturbed_perEpoch)) #range is end exclusive   
        i = 0
        for data, target, filename in tqdm(train_loader):
            #Data len is 16
            #Data[0] shape is [1, 128, 128]
            if i in batches_to_perturb:
                total_batches_perturbed += 1
                model.eval()
                perturbed_data = projected_gradient_descent(model, data, args['pjgd_eps'], 0.01, 100, np.inf)
                model.train()
                #torch.cat((perturbed_data, data), dim=0)
                #Perform Robust Training
                perturbed_data, target = perturbed_data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(perturbed_data)
                loss = loss_fn(out, target)
                loss.backward()
                train_loss += torch.sum(loss_fn(out, target)).item()
                correct += torch.sum(acc_fn(out.cpu(), target.cpu())).item()
                optimizer.step()
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            out = model(data)
            loss = loss_fn(out, target)
            loss.backward()
            train_loss += torch.sum(loss_fn(out, target)).item()
            correct += torch.sum(acc_fn(out.cpu(), target.cpu())).item()
            optimizer.step()
            i += 1

        train_loss /= len(train_loader.dataset)
        print('\nTrain set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

        test(model, device, test_loader, loss_fn, acc_fn)
        model.train()

def main(
    batch_size: Optional[int] = typer.Option(16, help='Input batch size for training (default: 64).'), 
    epochs: Optional[int] = typer.Option(10, help='Number of epochs to train (default: 15).'), 
    lr: Optional[float] = typer.Option(2e-4, help='Learning rate (default: 0.1).'), 
    pjgd_training: Optional[int] = typer.Option(10, help='Percent of pjgd perturbed data eps included in training set'),
    pjgd_eps: Optional[int] = typer.Option(.01, help='Pjgd eps'),
    fgsm_training: Optional[int] = typer.Option(0, help='Percent of fgsm perturbed data included in training set'),
    fgsm_eps: Optional[int] = typer.Option(0, help='Fgsm eps'),
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
        'pjgd_training': pjgd_training,
        'pjgd_eps': pjgd_eps,
        'fgsm_training': fgsm_training,
        'fgsm_eps': fgsm_eps
    }
    torch.manual_seed(seed)
    print(batch_size, " batch size")
    train_loader, test_loader = load_single_data(batch_size)
    
    #Append perturbed data
    

    model = CNNModel_128().to(device)
    #model = VitModel.to(device)
   
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = lambda out, target: nn.functional.one_hot(torch.argmax(out, dim=1), 10) * target

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(args, model, device, train_loader, test_loader, optimizer, loss_fn, acc_fn)

    if save_model:
        torch.save(model.state_dict(), output_file)

    test(model, device, test_loader, loss_fn, acc_fn)

if __name__ == '__main__':
    typer.run(main)

    train_loader, test_loader = load_single_data(16)
    #normalize_epsilon(test_loader)