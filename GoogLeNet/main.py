import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import argparse
from train_test import train, test
from model import GoogLeNet



if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser(description="GoogLeNet for CIFAR-100 Training")
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model_path = 'googlenet_cifar100.ckpt'

    # CIFAR-100 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5088964127604166, 0.48739301317401956, 0.44194221124387256), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Learning rate and num_epochs values to loop over
    learning_rates = [0.01]
    num_epochs_list = [50]

    for learning_rate in learning_rates:
        for num_epochs in num_epochs_list:
            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

            # Model
            model = GoogLeNet(num_classes=100).to(device)

            # Train the model
            train(model, train_loader, num_epochs, learning_rate, device, model_path, epoch_start=1)

            # Load the best model
            model = GoogLeNet(num_classes=100).to(device)
            model.load_state_dict(torch.load(model_path))
            model.to(device)

            # Test the model
            test(model, test_loader, device, num_epochs, learning_rate)