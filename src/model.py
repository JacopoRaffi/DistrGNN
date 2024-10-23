import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

#TODO: add superpixel part

class HGNN(nn.Module):
    '''
    Implementation of the Hierarchical Graph Neural Network (HGNN) model from
    the paper "Long, Jianwu. "A graph neural network for superpixel image classification." Journal of Physics: Conference Series. Vol. 1871. No. 1. IOP Publishing, 2021."

    Attributes:
    ----------
    convi: torch_geometric.nn.GATConv
        GATConv layer i 
    '''
    
    def __init__(self):
        super(HGNN, self).__init__()

        # The hyperparameters are the same as the ones used in the original paper
        self.conv1 = GATConv(3, 320, heads=1)
        self.conv2 = GATConv(320, 320, heads=1)
        self.conv3 = GATConv(320, 320, heads=1)
        self.conv4 = GATConv(320, 320, heads=1)

    def forward(self, x, edge_index, batch):
        out1 = F.relu(self.conv1(x, edge_index))
        out2 = F.relu(self.conv2(out1, edge_index))
        out3 = F.relu(self.conv3(out2, edge_index))
        out4 = F.relu(self.conv4(out3 + out1, edge_index))

        concat_out = torch.concat((out1, out2, out3, out4), dim=1)
        x = global_add_pool(concat_out, batch) # compute the graph embedding

        return x

class Classifier(nn.Module):
    '''
    Classifier is the classifier head of the HGNN model. It takes the graph embedding as input and outputs the logits for each class.

    Attributes:
    ----------
    gnn: HGNN
        The hierarchical graph neural network model
    fci: torch.nn.Linear
        Fully connected layer i
    '''
    def __init__(self, embedding_size, hidden_size, output_size):
        super(Classifier, self).__init__()

        self.gnn = HGNN()

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # return the unnormalized logits (multiclassification task)
        return x 
    
# ----------------------------------------------------------------- Pipeline Model -----------------------------------------------------------------