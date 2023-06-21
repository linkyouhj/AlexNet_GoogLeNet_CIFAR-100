import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import csv

def train(model, train_loader, num_epochs, learning_rate, device, save_path, epoch_start):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=15, gamma=0.2)

    # Training
    total_step = len(train_loader)
    progress_bar = tqdm(total=total_step, desc=f"Epoch [{epoch_start}/{num_epochs}]")
    best_loss = float('inf')

    with open('output_aux.csv', 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Accuracy', 'Top-5 Error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for epoch in range(epoch_start, num_epochs+1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            top5_correct = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs, aux1, aux2 = model(images)
                loss = criterion(outputs, labels) + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Calculate top-5 error
                _, top5_predicted = torch.topk(outputs.data, 5, dim=1)
                top5_correct += labels.view(-1, 1).expand_as(top5_predicted).eq(top5_predicted).sum().item()

            accuracy = 100 * correct / total
            top5_error = 100 - (100 * top5_correct / total)

            progress_bar.set_postfix({"Loss": running_loss / (i + 1), "Accuracy": accuracy, "Top-5 Error": top5_error})
            progress_bar.update(1)

            progress_bar.set_description(f"Epoch [{epoch}/{num_epochs}]")

            # Save the epoch loss, accuracy, and top-5 error to the CSV file
            writer.writerow({'Epoch': epoch, 'Loss': running_loss / (i + 1), 'Accuracy': accuracy, 'Top-5 Error': top5_error})

            # Save the best model based on loss
            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(model.state_dict(), save_path)
            # Adjust learning rate
            scheduler.step()
    progress_bar.close()

def test(model, test_loader, device, epoch, learning_rate):
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate top-5 error
            _, top5_predicted = torch.topk(outputs.data, 5, dim=1)
            top5_correct += labels.view(-1, 1).expand_as(top5_predicted).eq(top5_predicted).sum().item()

    accuracy = 100 * correct / total
    top5_error = 100 - (100 * top5_correct / total)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Top-5 Error: {top5_error:.2f}%")

    # Save test accuracy and top-5 error to the CSV file
    with open('output_1_aux.csv', 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Learning Rate', 'Test Accuracy', 'Top-5 Error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Epoch': epoch, 'Learning Rate': learning_rate, 'Test Accuracy': accuracy, 'Top-5 Error': top5_error})
