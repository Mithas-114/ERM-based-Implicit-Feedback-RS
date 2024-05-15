import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
    
class RatingRecoverModel(nn.Module):
    
    def __init__(self, net, train, test=None, device="cpu"):
        super(RatingRecoverModel, self).__init__()
        self.net = net.to(device)
        self.trainset = train
        self.testset = test
        self.device = device
    
    def fit(self, epochs=10, lr=1e-3, weight_decay=1e-2, alpha=0, patience=3):

        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        Loss = nn.MSELoss(reduction="mean")
        train_loss_list = []

        if self.testset is not None: 
            best_test_evaluation = torch.inf
            cur_patience = 0 

        for epoch in range(epochs):
            train_loss = 0
            for batch in tqdm(self.trainset):
                batch_size = len(batch)
                u_idx = batch[:, 0].to(self.device, dtype=torch.int)
                i_idx = batch[:, 1].to(self.device, dtype=torch.int)
                label = batch[:, 2].to(self.device, dtype=torch.float32)
                output = self.net(u_idx, i_idx)
                loss = Loss(output, label) + alpha*self.net.L2(u_idx, i_idx)/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(self.trainset))
        
            if self.testset is not None:
                with torch.no_grad():
                    evaluation = self.evaluate()
                    if evaluation > best_test_evaluation:
                        cur_patience += 1
                    else:
                        cur_patience = 0
                        best_test_evaluation = evaluation
                    if cur_patience > patience:
                        break
        plt.plot(train_loss_list)
        if self.testset is not None:
            return best_test_evaluation
        
    def evaluate(self):
        mse_loss = 0
        Loss = nn.MSELoss(reduction="mean")
        for batch in tqdm(self.testset):
            u_idx = batch[:, 0].to(self.device, dtype=torch.int)
            i_idx = batch[:, 1].to(self.device, dtype=torch.int)
            label = batch[:, 2].to(self.device, dtype=torch.float32)
            output = self.net(u_idx, i_idx)
            mse_loss += Loss(output, label).item()
        mse_loss /= len(self.testset)
        return mse_loss
    
    def get_rating(self):
        r = torch.clip(self.net.U.matmul(self.net.I.T), 1, 5)
        return r
    

class ObservationModel(nn.Module):
    
    def __init__(self, net, train, test=None, device="cpu"):
        super(ObservationModel, self).__init__()
        self.net = net.to(device)
        self.trainset = train
        self.testset = test
        self.device = device
    
    def fit(self, epochs=10, lr=1e-3, weight_decay=1e-2, alpha=0, patience=3):

        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        Loss = nn.BCELoss(reduction="mean")
        train_loss_list = []

        if self.testset is not None: 
            best_test_acc = 0
            cur_patience = 0 

        for epoch in range(epochs):
            train_loss = 0
            for batch in tqdm(self.trainset):
                batch_size = len(batch)
                u_idx = batch[:, 0].to(self.device, dtype=torch.int)
                i_idx = batch[:, 1].to(self.device, dtype=torch.int)
                label = batch[:, 2].to(self.device, dtype=torch.float32)
                output = torch.sigmoid(self.net(u_idx, i_idx))
                loss = Loss(output, label) + alpha*self.net.L2(u_idx, i_idx)/batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss_list.append(train_loss/len(self.trainset))
        
            if self.testset is not None:
                with torch.no_grad():
                    acc = self.evaluate()
                    if acc <= best_test_acc:
                        cur_patience += 1
                    else:
                        cur_patience = 0
                        best_test_acc = acc
                    if cur_patience > patience:
                        break
        plt.plot(train_loss_list)
        if self.testset is not None:
            return best_test_acc
        
    def evaluate(self):
        acc = 0
        for batch in tqdm(self.testset):
            u_idx = batch[:, 0].to(self.device, dtype=torch.int)
            i_idx = batch[:, 1].to(self.device, dtype=torch.int)
            label = batch[:, 2].to(self.device, dtype=torch.int)
            output = torch.sigmoid(self.net(u_idx, i_idx))
            prediction = output >= 1/2
            acc += (prediction==label).to(dtype=torch.float32).mean()
        acc /= len(self.testset)
        return acc
    
    def get_theta(self):
        theta = torch.sigmoid(self.net.U.matmul(self.net.I.T))
        return theta
    



            
