import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from ..utils.data import split_train_and_val, get_train_data
from ..models.models import BasicMF
from ..utils.metrics import *

class weighted_MSELoss(nn.Module):
    def __init__(self):
        super(weighted_MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")
    
    def forward(self, inputs, targets, weight):
        """
        inputs : 预测值
        targets : Y
        weight是单个值
        根据WMF，inputs = 0的需要增加weight
        """
        weights = torch.ones(len(inputs)).to("cuda:0")
        weights[targets==0] = weight
        loss_list = self.loss(inputs, targets)
        loss = torch.mean(loss_list*weights)
        return loss
    
class weighted_BCELoss(nn.Module):
    def __init__(self):
        super(weighted_BCELoss, self).__init__()
        self.loss = nn.BCELoss(reduction="none")
    
    def forward(self, inputs, targets, weight):
        """
        inputs : 预测值
        targets : Y
        weight是单个值
        根据WMF，inputs = 0的需要增加weight
        """
        weights = targets + (1-targets)*weight
        loss_list = self.loss(inputs, targets)
        loss = torch.mean(loss_list*weights)
        return loss
    
def Train(train, val, model, model_name, test=None, **kwargs):
    """
    train, val : dataloader，分别为训练集、验证集
    test : 测试集，可以不需要
    model : 训练网络
    model_name : 模型名称('WMF', 'REL-MF', 'ERM-MF-PRIOR', 'ERM-MF-ESTIMATE')
    epochs : 训练次数
    lr : 学习率
    alpha : L2损失
    patience : 容忍度
    """
    device = kwargs["device"]
    if model_name == "WMF":
        # 使用MSELoss
        Loss = weighted_MSELoss()
    else:
        # 使用BCELoss
        Loss = nn.BCELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=kwargs["lr"], weight_decay=5e-4)
    schedule = ReduceLROnPlateau(optimizer,'max', factor=0.1, patience=2, verbose=True)

    train_loader = DataLoader(train, batch_size=8192,shuffle=True)
    val_loader = DataLoader(val, batch_size=8192,shuffle=False) 
    best_val_score = 0
    cur_patience = 0

    for epoch in range(kwargs["epochs"]):
        train_loss = 0
        print(f"------ epoch:{epoch+1} ------")
        for batch in tqdm(train_loader):
            u_idx = batch[:, 0].to(device, dtype=torch.int)
            i_idx = batch[:, 1].to(device, dtype=torch.int)
            if model_name == "WMF":
                # user | item | Y | IPS
                label = batch[:, 2].to(device, dtype=torch.float32)
                output = model(u_idx, i_idx)
                loss = Loss(output, label, kwargs["weight"]) 
            else:
                # user | item | Y | IPS |...| label 
                label = batch[:, -1].to(device, dtype=torch.float32)
                output = torch.sigmoid(model(u_idx, i_idx))
                loss = Loss(output, label)
            loss += kwargs["alpha"]*model.L2(u_idx, i_idx)/len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                u_idx = batch[:, 0].to(device, dtype=torch.int)
                i_idx = batch[:, 1].to(device, dtype=torch.int)
                # val的loss不加正则项，但是与train使用相同的模型
                if model_name == "WMF":
                    # user | item | Y | IPS
                    label = batch[:, 2].to(device, dtype=torch.float32)
                    output = model(u_idx, i_idx)
                    loss = Loss(output, label, kwargs["weight"]) 
                else:
                    # user | item | Y | IPS |...| label 
                    label = batch[:, -1].to(device, dtype=torch.float32)
                    output = torch.sigmoid(model(u_idx, i_idx))
                    loss = Loss(output, label)
                val_loss += loss.item()  
            val_loss /= len(val_loader)
        val_score = Evaluate(val, model, use_IPS=True, metrics={"RECALL":cal_RECALL}, k_list=[5], **kwargs).loc["RECALL", 5]
        print("Epoch %s: train loss : %10.6f   val loss : %10.6f   val score : %10.6f"%(epoch+1, train_loss,  val_loss, val_score))
                        
        schedule.step(val_score)
        if val_score <= best_val_score:
            cur_patience += 1
            if cur_patience > kwargs["patience"]:
                break
        else:
            cur_patience = 0
            best_val_score = val_score
            
    if test is not None:
        test_loader = DataLoader(test, batch_size=8192 ,shuffle=False)
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                u_idx = batch[:, 0].to(device, dtype=torch.int)
                i_idx = batch[:, 1].to(device, dtype=torch.int)
                if model_name == "WMF":
                    # user | item | Y | IPS
                    # user | item | R
                    label = batch[:, 2].to(device, dtype=torch.float32)
                    output = model(u_idx, i_idx)
                    loss = Loss(output, label, kwargs["weight"]) 
                else:
                    # user | item | Y | IPS |...| label 
                    # user | item | R
                    label = batch[:, -1].to(device, dtype=torch.float32)
                    output = torch.sigmoid(model(u_idx, i_idx))
                    loss = Loss(output, label)
                test_loss += loss.item()    
            test_loss /= len(test_loader)
        return test_loss


def Evaluate(data, model, use_IPS=True, k_list=[10], metrics = {"MAP":cal_AP, "NDCG": cal_NDCG, "RECALL": cal_RECALL}, **kwargs):
    """
    user | item | Y | IPS | ...
    user | item | R 
    """
    results = defaultdict(list)
    users = data[:, 0].to(dtype=torch.int64)
    
    for user in tqdm(set(users.detach().cpu().numpy())):
        user = torch.tensor(user, dtype=torch.int64) 
        indices = (users==user)
        items = data[indices, 1].to(dtype=torch.int64) # 该user对应的items
        IPS = data[indices, 3] if use_IPS else None # 该user对应的IPS
        true_scores = data[indices, 2] # 该user的Y
        scores = model(user, items).detach().cpu() # 该user的scores
        for k in k_list :
            for metric, metric_func in metrics.items():
                results[f"{metric}@{k}"].append(metric_func(true_scores, scores, k, IPS).detach().cpu().numpy())
                
    results = {key:np.mean(np.array(value_list)) for key, value_list in results.items()}
    results_df = pd.DataFrame(np.zeros([len(metrics), len(k_list)])).rename(index={i:list(metrics.keys())[i] for i in range(len(metrics))},
                                                                           columns={i:k_list[i] for i in range(len(k_list))})
    for k in k_list:
        for metric in metrics.keys():
            results_df.loc[metric, k] = results[f"{metric}@{k}"]

    return results_df


def ParamSelection(data, model_name, task=1, k=4, **kwargs):
    """
    参数选取
    每一轮中，data依据LOO划分为不相交的train, val_in, val_out. val_in 作为train训练时的
    验证集，val_out作为测试集，均使用SNIPS度量和RECALL@5.  
    """
    Scores = []
    Log_loss = []
    for i in range(k):
        print(f"------ 第{i+1}次实验 ------")
        train, val_in, val_out = split_train_and_val(data, kwargs["num_users"], kwargs["num_items"], need_val_out=True)
        # 定义模型，开始训练
        model = BasicMF(num_users=kwargs["num_users"], 
                        num_items=kwargs["num_items"], 
                        embedding_dim=kwargs["embedding_dim"]).to(kwargs["device"])
        log_loss = Train(train, val_in, test=val_out, model=model, model_name=model_name, **kwargs)
        score = Evaluate(val_out, model, use_IPS=True, metrics={"RECALL":cal_RECALL}, k_list=[5], **kwargs).loc["RECALL", 5]
        Log_loss.append(log_loss)
        Scores.append(score)
        print(f"------ log loss of {i+1} fold : {log_loss} ------")
        print(f"------ score of {i+1} fold : {score} ------")
    print("parameters : ", kwargs)
    print(f"------ log loss of the model : {np.array(Log_loss).mean()} ------")
    print(f"------ score of the model : {np.array(Scores).mean()} ------")
    # 保存
    if not os.path.exists(f"ParamSelectionRecord-task{task}/{model_name}"):
        os.makedirs(f"ParamSelectionRecord-task{task}/{model_name}")
    params = {
        'log_loss':np.array(Log_loss).mean(),
        'score':np.array(Scores).mean(),
        'model name':model_name}
    params.update(kwargs)
    files = os.listdir(f"ParamSelectionRecord-task{task}/{model_name}")
    idx = len(files) + 1
    save_path = f"ParamSelectionRecord-task{task}/{model_name}/params-{idx}.json"
    with open(save_path, "w") as f:
        json.dump(params, f)


def Test(train, test, model_name, task=1, repeats=10, **kwargs):
    """
    测试
    每一轮中，data依据LOO划分为不相交的train, val. val作为train训练时的
    验证集, 使用SNIPS度量和RECALL@5.  test 使用全部标准度量。
    """
    test_score = []
    test_log_loss = []
    for repeat in range(repeats):
        print(f"------ repeat : {repeat+1} ------")
        train_i, val_i = split_train_and_val(train, kwargs["num_users"], kwargs["num_items"], need_val_out=False)
        model = BasicMF(num_users=kwargs["num_users"], 
                        num_items=kwargs["num_items"], 
                        embedding_dim=kwargs["embedding_dim"]).to(kwargs["device"])
        test_log_loss_i = Train(train_i, val_i, test=test, model=model, model_name=model_name, **kwargs)
        test_score_i = Evaluate(test, model, use_IPS=False, metrics={"NDCG":cal_NDCG, "MAP":cal_AP, "RECALL":cal_RECALL}, k_list=list(range(1, 6)), **kwargs).values
        test_score.append(test_score_i)
        test_log_loss.append(test_log_loss_i)
    test_score_mean = np.mean([test_score_i for test_score_i in test_score], axis=0)
    test_score_std = np.std([test_score_i for test_score_i in test_score], axis=0)
    mean_df = pd.DataFrame(np.zeros([3, 5])).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(5)})
    std_df = pd.DataFrame(np.zeros([3, 5])).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(5)})
    for i in range(3):
        for j in range(5):
            mean_df.iloc[i, j] = test_score_mean[i, j]
            std_df.iloc[i, j] = test_score_std[i, j]
    # 保存
    if not os.path.exists(f"TestRecord-task{task}/{model_name}"):
        os.makedirs(f"TestRecord-task{task}/{model_name}")
    params = {
        'log_loss':np.array(test_log_loss).mean(),
        'model name':model_name}
    params.update(kwargs)
    files = os.listdir(f"TestRecord-task{task}/{model_name}")
    idx = len(files) + 1
    save_path = f"TestRecord-task{task}/{model_name}/params-{idx}.json"
    with open(save_path, "w") as f:
        json.dump(params, f)
    mean_df.to_csv(f"TestRecord-task{task}/{model_name}/mean.csv")
    std_df.to_csv(f"TestRecord-task{task}/{model_name}/std.csv")
    for i in range(repeats):
        test_score_i = pd.DataFrame(test_score[i]).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(5)})
        test_score_i.to_csv(f"TestRecord-task{task}/{model_name}/score-{i+1}.csv")

        
def computing_r_hat_using_ERM(addr, save_addr, kwargs):
    model = BasicMF(kwargs["num_users"], kwargs["num_items"], kwargs["embedding_dim"]).to("cuda:0")
    data = get_train_data(addr, kwargs["model_name"], phi_r=kwargs["phi_r"], phi_theta=kwargs["phi_theta"], 
                           lam=kwargs["lam"], mu=kwargs["mu"], r_default=kwargs["r_default"], K=kwargs["K"])
    train, val = split_train_and_val(data, kwargs["num_users"], kwargs["num_items"], need_val_out=False)
    Train(train, val, model_name=kwargs["model_name"], test=None, model=model, lr=kwargs["lr"], alpha=kwargs["alpha"], 
          patience=3, epochs=30, device="cuda:0")
    r_hat = torch.sigmoid(model(data[:, 0].to("cuda:0", dtype=torch.int), data[:, 1].to("cuda:0",dtype=torch.int))
                         ).detach().cpu().numpy()
    np.save(os.path.join(save_addr, "r_hat.npy"), r_hat)
    with open(os.path.join(save_addr, "params.json"), "w") as f:
        json.dump(kwargs, f)
    