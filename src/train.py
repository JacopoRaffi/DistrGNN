from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import csv
from torch_geometric.loader import DataLoader

from model import ClassifierHead
from data import CifarDataset, image_to_graph

#TODO: comment and test, add performance metrics (write to csv)
#TODO: change the 'print time' to 'log time' and write to a csv file (save data in list and then write to csv), so to avoid file writing overhead
#TODO: add also memory usage logging, use torch profile function

def train(model, optimizer, criterion, train_loader, val_loader, epoch, device):
    start_training_time = time.time() * 1000
    for epoch in range(epoch):
        epoch_start_time = time.time() * 1000
        model.train()
        total_loss = 0
        mean_batch_time = 0
        for data in train_loader:
            start_batch_time = time.time() * 1000
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            
            loss = criterion(output, data.y)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            end_batch_time = time.time() * 1000
            mean_batch_time += end_batch_time - start_batch_time
        print(f'Mean batch time: {mean_batch_time / len(train_loader)} ms')
        
        epoch_end_time = time.time() * 1000
        print(f'Epoch time: {epoch_end_time - epoch_start_time} ms')
        print(f'Epoch: {epoch}, Train Loss: {total_loss / len(train_loader)}')
        model.eval()

        with torch.no_grad():
            total_loss = 0
            mean_val_batch_time = 0
            for data in val_loader:
                start_val_batch_time = time.time() * 1000
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                total_loss += loss.item()
                end_val_batch_time = time.time() * 1000
                mean_val_batch_time += end_val_batch_time - start_val_batch_time
            print(f'Mean val batch time: {mean_val_batch_time / len(val_loader)} ms')

            print(f'Epoch: {epoch}, Val Loss: {total_loss / len(val_loader)}')

    end_training_time = time.time() * 1000
    print(f'Full Training time: {end_training_time - start_training_time} ms')
    return total_loss / len(train_loader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = ClassifierHead(1280, 512, 10).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    transform = T.ToTensor()

    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    train_dataset = CifarDataset(image_to_graph(train_dataset))
    test_dataset = CifarDataset(image_to_graph(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    train(gnn, optimizer, criterion, train_loader, val_loader, 10, device)