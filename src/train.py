from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import csv

from model import ClassifierHead
from data import CifarDataset, image_to_graph

#TODO: comment and test, add performance metrics (write to csv)

def train(model, optimizer, criterion, train_loader, val_loader, epoch, device):
    
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            
            loss = criterion(output, data.y)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        print(f'Epoch: {epoch}, Train Loss: {total_loss / len(train_loader)}')

        with torch.no_grad():
            total_loss = 0
            
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                total_loss += loss.item()

            print(f'Epoch: {epoch}, Val Loss: {total_loss / len(val_loader)}')


    return total_loss / len(train_loader)


if __name__ == '__main__':
    gnn = ClassifierHead(1280, 512, 10)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = T.ToTensor()

    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    train_dataset = CifarDataset(image_to_graph(train_dataset))
    test_dataset = CifarDataset(image_to_graph(test_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    train(gnn, optimizer, criterion, train_loader, val_loader, 10, device)