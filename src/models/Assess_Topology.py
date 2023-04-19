# Import modules
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd.variable import Variable
from src.topology import PersistentHomologyCalculation
from src.models.approx_based import TopologicalSignatureDistance

# Define function to compute distance matrix
def compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    return distances

def compute_topological_error(data_01, data_02):
    topo_sig = TopologicalSignatureDistance(match_edges='symmetric')

    distances_01 = compute_distance_matrix(data_01)
    distances_02 = compute_distance_matrix(data_02)

    topo_error, topo_error_components = topo_sig(distances_01, distances_02)

    return topo_error
