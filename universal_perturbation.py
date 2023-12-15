import trainer as trainer_module
import matplotlib.pyplot as plt
import adversarial_perturbation
from preprocess import load_single_data
import torch
def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    trainer = trainer_module.trainer()
    #trainset,testset  = data_loader.load_data()
    train_loader, test_loader = load_single_data(16, False)   #Set true if ViT
    print("loaded data")
    #accuracy = trainer.train(train_loader, test_loader)
    #print("trainer complete")
    train_loader, test_loader = load_single_data(16, False)   #Set true if ViT
    model = trainer.net.to(device)

   
    v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(.83, train_loader, test_loader, model)
    print("Perturbation ran, generating graphs")
    plt.title("Fooling Rates over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Fooling Rate on test data")
    plt.plot(total_iterations,fooling_rates)
    plt.show()


    plt.title("Accuracy over Universal Iterations")
    plt.xlabel("Universal Algorithm Iter")
    plt.ylabel("Accuracy on Test data")
    plt.plot(total_iterations, accuracies)
    plt.show()



if __name__ == "__main__":
    main()