from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import torch
import time
import argparse
import csv
from torch_geometric.loader import DataLoader

from model import HGNN
from data import CustomDataset, image_to_graph

#TODO: add also memory usage logging, use torch profile function

def train(model, optimizer, criterion, train_loader, val_loader, epoch, device, filename):
    '''
    Train the model and compute the performance metrics

    Parameters:
    ----------
    model: torch.nn.Module
        Model to train
    optimizer: torch.optim.Optimizer
        Optimizer to use
    criterion: torch.nn.Module
        Loss function
    train_loader: torch_geometric.loader.DataLoader
        DataLoader for training
    val_loader: torch_geometric.loader.DataLoader
        DataLoader for validation (test)
    epoch: int
        Number of epochs
    device: torch.device
        Device to use
    filename: str
        Name of the log file where to store time metrics
        

    Returns:
    -------
    return: None
    '''

    # Log the training and validation (test) time
    with open(filename, 'w+') as log_file: 
        csv_writer = csv.writer(log_file)
        header = ['epoch', 'batch', 'batch_time(s)', 'phase'] # Phase: 0 - train, 1 - val
        csv_writer.writerow(header)
        log_file.flush()

        for epoch in range(epoch):
            model.train()
            total_loss = 0
            for i, data in enumerate(train_loader):
                csv_row = []
                csv_row.append(epoch)
                csv_row.append(i)
                
                start_batch_time = time.time()
                
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                end_batch_time = time.time()
                
                csv_row.append(end_batch_time - start_batch_time)
                csv_row.append(0)
                csv_writer.writerow(csv_row) # The row contains the epoch_id, the batch_id, the time spent in the batch and the phase
            
            model.eval()
            with torch.no_grad():
                total_loss = 0
                for i, data in enumerate(val_loader):
                    csv_row = []
                    csv_row.append(epoch)
                    csv_row.append(i)
                    
                    start_batch_time = time.time()
                    
                    data = data.to(device)
                    output = model(data.x, data.edge_index, data.batch)
                    loss = criterion(output, data.y)
                    total_loss += loss.item()
                    
                    end_batch_time = time.time()

                    csv_row.append(end_batch_time - start_batch_time)
                    csv_row.append(1)
                    csv_writer.writerow(csv_row)

            log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='Name of the log file', default='tmp.csv')
    parser.add_argument('-l', type=int, help='Length of the dataset to consider', default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gnn = HGNN(1280, 512, 10).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    transform = T.ToTensor() #TODO: add the super pixel part

    train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=False, transform=transform)

    train_dataset = CustomDataset(image_to_graph(train_dataset), length=args.l)
    test_dataset = CustomDataset(image_to_graph(test_dataset), length=args.l)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    train(gnn, optimizer, criterion, train_loader, val_loader, 5, device, args.filename)