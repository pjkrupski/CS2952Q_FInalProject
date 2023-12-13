from models import CNNModel_128
import torch
from torch import nn
from torch import optim

device = torch.device('cpu')

class trainer:
    def __init__(self):


        #self.net = model.ConvNet()
        self.net = CNNModel_128()
        self.net.to(device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.n_epochs = 4

    def train(self,trainloader,testloader):
        accuracy = 0
        self.net.train()
        for epoch in range(self.n_epochs):
            #TODO: Remove when finished debugging
            #Save accuracy and pass as constant in universal_perturbations.py
            break
            running_loss = 0.0
            print_every = 200  # mini-batches
            #Added filename to fit trainloader shape
            for i, (inputs, labels, filename) in enumerate(trainloader, 0):
                # Transfer to GPU
                print("training...", i, "/", len(trainloader))
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % print_every) == (print_every-1):
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/print_every))
                    running_loss = 0.0


            # Print accuracy after every epoch
            accuracy = compute_accuracy(self.net, testloader)
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))

        print('Finished Training')
        return 0 #accuracy



def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        i = 0
        for images, labels, filenames in testloader:
            print("testing ", i, "/", len(testloader))
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            i += 1
    print("test complete")
    return correct / total



