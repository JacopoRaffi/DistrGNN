import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class HGNN(nn.Module):
    '''
    Implementation of the Hierarchical Graph Neural Network (HGNN) model from
    the paper "Long, Jianwu. "A graph neural network for superpixel image classification." Journal of Physics: Conference Series. Vol. 1871. No. 1. IOP Publishing, 2021."

    Attributes:
    ----------
    convi: torch_geometric.nn.GATConv
        GATConv layer i 
    fci: torch.nn.Linear
        Fully connected layer i
    '''
    
    def __init__(self, embedding_size, hidden_size, output_size):
        super(HGNN, self).__init__()

        # The hyperparameters are the same as the ones used in the original paper
        self.conv1 = GATConv(3, 320, heads=1)
        self.conv2 = GATConv(320, 320, heads=1)
        self.conv3 = GATConv(320, 320, heads=1)
        self.conv4 = GATConv(320, 320, heads=1)

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, batch):
        out1 = F.relu(self.conv1(x, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv4(out3 + out1, edge_index))

        concat_out = torch.concat((out1, out2, out3, out4), dim=1)
        graph_embedding = global_add_pool(concat_out, batch) # compute the graph embedding

        graph_embedding = F.relu(self.fc1(graph_embedding))
        graph_embedding = self.fc2(graph_embedding)

        return graph_embedding