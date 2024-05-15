import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BasicMF(nn.Module):
    """
    the basic matrix factorization model
    """
    
    def __init__(self, num_users, num_items, embedding_dim):
        super(BasicMF, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.U = nn.Parameter(torch.randn(self.num_users, self.embedding_dim))
        self.I = nn.Parameter(torch.randn(self.num_items, self.embedding_dim))
        self.user_bias = nn.Parameter(torch.randn(self.num_users))
        self.item_bias = nn.Parameter(torch.randn(self.num_items))
        self.global_bias = nn.Parameter(torch.tensor(1e-3))
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.I)
        
    def forward(self, u_idx, i_idx):
        vu = self.U[u_idx, :]
        vi = self.I[i_idx, :]
        output = torch.sum(vu*vi, dim=1) \
            + self.user_bias[u_idx] + self.item_bias[i_idx] + self.global_bias 
        return output
    
    def L2(self, u_idx, i_idx):
        return torch.sum(self.U[u_idx, :]**2) + torch.sum(self.I[i_idx, :]**2) + torch.sum(self.user_bias[u_idx]**2) + torch.sum(self.item_bias[i_idx]**2) + self.global_bias**2
    