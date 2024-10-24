from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import time
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.loader import DataLoader

from data import CustomDataset, image_to_graph


transform = T.ToTensor()
train_dataset = CIFAR10(root='../data', train=True, download=False, transform=transform)
dataset = CustomDataset(image_to_graph(train_dataset), length=150)
loader = DataLoader(dataset, batch_size=50, shuffle=True)

gat_1 = GATConv(3, 320, heads=1)
gat_2 = GATConv(320, 320, heads=1)
linear_1 = torch.nn.Linear(1280, 512)
linear_2 = torch.nn.Linear(512, 10)

df_empty ={'layer' : [],
            'time(s)' : []}

iterator = iter(loader)
N = 3

for i in range(N):
    data = next(iterator)
    
    start_time = time.time()
    out1 = F.relu(gat_1(data.x, data.edge_index))
    end_time = time.time()

    df_empty['layer'].append('GATConv1')
    df_empty['time(s)'].append(end_time - start_time)

    start_time = time.time()
    out2 = F.relu(gat_2(out1, data.edge_index))
    end_time = time.time()

    df_empty['layer'].append('GATConv2')
    df_empty['time(s)'].append(end_time - start_time)

    start_time = time.time()
    concat_out = torch.concat((out1, out2, out2, out2), dim=1)
    graph_embedding = global_add_pool(concat_out, data.batch) # compute the graph embedding
    end_time = time.time()

    df_empty['layer'].append('Concat+Pooling')
    df_empty['time(s)'].append(end_time - start_time)

    start_time = time.time()
    linear_out = F.relu(linear_1(graph_embedding))
    end_time = time.time()

    df_empty['layer'].append('Linear1')
    df_empty['time(s)'].append(end_time - start_time)

    start_time = time.time()
    logits = linear_2(linear_out)
    loss = F.cross_entropy(logits, data.y)
    end_time = time.time()

    df_empty['layer'].append('Linear2+Loss')
    df_empty['time(s)'].append(end_time - start_time)

pd.DataFrame(df_empty).to_csv('../log/layer.csv', index=False)