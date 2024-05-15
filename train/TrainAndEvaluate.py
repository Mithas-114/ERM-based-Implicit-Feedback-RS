import os
import json
import torch
import urllib.parse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from utils.data import *
from models.model import BasicMF
from utils.metrics import *


def Train(train, val, test=None, model=None, **kwargs):
    """
    train, val : dataloader，分别为训练集、验证集
    model : 训练网络
    epochs : 训练次数
    lr : 学习率
    alpha : L2损失
    patience : 容忍度
    """
      
    Loss = nn.BCELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=kwargs["lr"], weight_decay=5e-4)
    schedule = ReduceLROnPlateau(optimizer,'max', factor=0.1, patience=2, verbose=True)

    train_loader = DataLoader(train, batch_size=2**12,shuffle=True)
    val_loader = DataLoader(val, batch_size=2**12,shuffle=False)
    
    best_val_score = 0
    cur_patience = 0 
    device = kwargs["device"]
    
    data_for_scoring = get_data_for_computing_score(val)
        
    for epoch in range(kwargs["epochs"]):
        # 一轮训练
        train_loss = 0
        print(f"------ epoch:{epoch+1} ------")
        for batch in tqdm(train_loader):
            u_idx = batch[:, 0].to(device, dtype=torch.int)
            i_idx = batch[:, 1].to(device, dtype=torch.int)
            label = batch[:, -1].to(device, dtype=torch.float32)
            output = torch.sigmoid(model(u_idx, i_idx))
            loss = Loss(output, label) + kwargs["alpha"]*model.L2(u_idx, i_idx)/len(batch)
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
                label = batch[:, -1].to(device, dtype=torch.float32)
                # val的loss不加正则项，但是与train使用相同的模型
                output = torch.sigmoid(model(u_idx, i_idx))
                loss = Loss(output, label)
                val_loss += loss.item()    
            val_loss /= len(val_loader)

        val_score = Evaluate(data_for_scoring, model, metrics={"RECALL":cal_RECALL}, k_list=[10], **kwargs).loc["RECALL", 10]

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
        test_loader = DataLoader(test, batch_size=2**12,shuffle=False)
        test_loss = 0
        with torch.no_grad():

            for batch in tqdm(test_loader):
                u_idx = batch[:, 0].to(device, dtype=torch.int)
                i_idx = batch[:, 1].to(device, dtype=torch.int)
                # 如果是交叉验证内部测试，则用相同标签。如果是外部测试，则用真实标签R
                if kwargs["model_selection"] == True:
                    label = batch[:, -1].to(device, dtype=torch.float32)
                else:
                    label = batch[:, 4].to(device, dtype=torch.float32)
                output = torch.sigmoid(model(u_idx, i_idx))
                loss = Loss(output, label)
                test_loss += loss.item()    
            test_loss /= len(test_loader)
            return test_loss


def Evaluate(test_data, model, k_list=[2,4,6,8,10], metrics = {"MAP":cal_AP, "NDCG": cal_NDCG, "RECALL": cal_RECALL}, **kwargs):
    
    results = defaultdict(list)
    
    for user, data in tqdm(test_data.items()):
        user = torch.tensor(user, dtype=torch.int)
        items = data[:, 1].to(dtype=torch.int)
        true_scores = data[:, 4]
        scores = model(user, items).detach().cpu()
        for k in k_list :
            for metric, metric_func in metrics.items():
                results[f"{metric}@{k}"].append(metric_func(true_scores, scores, k).detach().cpu().numpy())
                
    results = {key:np.mean(np.array(value_list)) for key, value_list in results.items()}
    results_df = pd.DataFrame(np.zeros([len(metrics), len(k_list)])).rename(index={i:list(metrics.keys())[i] for i in range(len(metrics))},
                                                                           columns={i:k_list[i] for i in range(len(k_list))})
    for k in k_list:
        for metric in metrics.keys():
            results_df.loc[metric, k] = results[f"{metric}@{k}"]

    return results_df


def CrossValidation(data, power, model_name, task, k=5, **kwargs):
    """
    参数选取的K-折交叉验证
    """
    if task in ["ideal", "cold user", "ablation", "varying K"]:
        data = get_train_data_with_ideal_theta(data, model_name, **kwargs) # user | item | ... |label
    elif task in ["theta hat", "IPS"]:
        data = get_train_data_with_theta_hat(data, model_name, **kwargs) # user | item | ... |label
    else:
        raise ValueError()
    # K-折划分
    data_split = list(range(k))
    ratio = 1/k
    for i in range(k-1):
        data, data_split[i] = train_test_split(data, test_size=1/(k-i), random_state=0)
    data_split[k-1] = data
    # 训练
    Scores = []
    Log_loss = []
    models = []
    for i in range(k):
        print(f"------ 第{i+1}折 ------")
        # 获取训练数据
        train = torch.vstack([data_split[j] for j in range(k) if j != i])       
        test = data_split[i]
        train, val = train_test_split(train, test_size=0.1, random_state=0)
        data_for_scoring = get_data_for_computing_score(test)
        
        # 定义模型，开始训练
        model = BasicMF(num_users=kwargs["num_users"], 
                        num_items=kwargs["num_items"], 
                        embedding_dim=kwargs["embedding_dim"]).to(kwargs["device"])
        log_loss = Train(train, val, test=test, model=model, **kwargs)
        score = Evaluate(data_for_scoring, model, metrics={"RECALL":cal_RECALL}, k_list=[10], **kwargs).loc["RECALL", 10]
        Log_loss.append(log_loss)
        Scores.append(score)
        models.append(model)
        print(f"------ log loss of {i+1} fold : {log_loss} ------")
        print(f"------ score of {i+1} fold : {score} ------")
    print(f"settings : power = {power}, model name = {model_name}")
    print("parameters : ", kwargs)
    print(f"------ log loss of the model : {np.array(Log_loss).mean()} ------")
    print(f"------ score of the model : {np.array(Scores).mean()} ------")
    # 保存
    if task in ["ideal", "cold user", "ablation", "IPS"]:
        if not os.path.exists(f"CrossValidationRecord/{task}/power={power}/{model_name}"):
            os.makedirs(f"CrossValidationRecord/{task}/power={power}/{model_name}")
        params = {
            'log_loss':np.array(Log_loss).mean(),
            'score':np.array(Scores).mean(),
            'power':power,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"CrossValidationRecord/{task}/power={power}/{model_name}")
        idx = len(files) + 1
        save_path = f"CrossValidationRecord/{task}/power={power}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)
    elif task == "theta hat":
        n = kwargs["n"]
        if not os.path.exists(f"CrossValidationRecord/{task}/power={power}/n={n}/{model_name}"):
            os.makedirs(f"CrossValidationRecord/{task}/power={power}/n={n}/{model_name}")
        params = {
            'log_loss':np.array(Log_loss).mean(),
            'score':np.array(Scores).mean(),
            'n':n,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"CrossValidationRecord/{task}/power={power}/n={n}/{model_name}")
        idx = len(files) + 1
        save_path = f"CrossValidationRecord/{task}/power={power}/n={n}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)   
    elif task == "varying K":
        K = kwargs["K"]
        metric = kwargs["similarity_metric"]
        similarity_of = urllib.parse.quote(kwargs["similarity_method"], safe='')
        if not os.path.exists(f"CrossValidationRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}"):
            os.makedirs(f"CrossValidationRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}")
        params = {
            'log_loss':np.array(Log_loss).mean(),
            'score':np.array(Scores).mean(),
            'K':K,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"CrossValidationRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}")
        idx = len(files) + 1
        save_path = f"CrossValidationRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)   
    
def computing_r_hat_using_ERM(data, task, kwargs, save_path=None):
    model = BasicMF(kwargs["num_users"], kwargs["num_items"], kwargs["embedding_dim"]).to("cuda:0")
    if task in ["cold user", "ideal", "ablation"] : 
        train = get_train_data_with_ideal_theta(data, model_name="ERM-MF-PRIOR", phi_r=kwargs["phi_r"], lam=kwargs["lam"])
    else:
        train = get_train_data_with_theta_hat(data,  model_name="ERM-MF-PRIOR", phi_r=kwargs["phi_r"], lam=kwargs["lam"],
                                             phi_theta=kwargs["phi_theta"], mu=kwargs["mu"])
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    Train(train, val, test=None, model=model, lr=kwargs["lr"], alpha=kwargs["alpha"], patience=3, epochs=30, device="cuda:0")
    r_hat = torch.sigmoid(model(data[:, 0].to("cuda:0", dtype=torch.int), data[:, 1].to("cuda:0", dtype=torch.int))).detach().cpu().reshape([-1, 1])
    if save_path:
        np.save(save_path, r_hat.numpy().flatten())
    return torch.hstack([data, r_hat])


def Test(train, test, power, model_name, task, repeats=10, **kwargs):
    test_score = []
    test_log_loss = []
    if model_name == "ERM-MF-ESTIMATE":
        train = computing_r_hat_using_ERM(train, task, kwargs["kwargs_"])
    if task == "cold user":
        test = get_data_for_cold_start_testing(train, test)
    for repeat in range(repeats):
        print(f"------ repeat : {repeat+1} ------")
        if task in ["ideal", "cold user", "ablation", "varying K"]:
            train_i = get_train_data_with_ideal_theta(train, model_name, **kwargs) # user | item | ... |label
        else:
            train_i = get_train_data_with_theta_hat(train, model_name, **kwargs) # user | item | ... |label
        train_i, val_i = train_test_split(train_i, test_size=0.2, random_state=0)
        data_for_scoring_i = get_data_for_computing_score(test)
        model = BasicMF(num_users=kwargs["num_users"], 
                        num_items=kwargs["num_items"], 
                        embedding_dim=kwargs["embedding_dim"]).to(kwargs["device"])
        test_log_loss_i = Train(train_i, val_i, test=test, model=model, **kwargs)
        test_score_i = Evaluate(data_for_scoring_i, model, metrics={"NDCG":cal_NDCG, "MAP":cal_AP, "RECALL":cal_RECALL}, k_list=list(range(1, 11)), **kwargs).values
        test_score.append(test_score_i)
        test_log_loss.append(test_log_loss_i)
    test_score_mean = np.mean([test_score_i for test_score_i in test_score], axis=0)
    test_score_std = np.std([test_score_i for test_score_i in test_score], axis=0)
    mean_df = pd.DataFrame(np.zeros([3, 10])).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(10)})
    std_df = pd.DataFrame(np.zeros([3, 10])).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(10)})
    for i in range(3):
        for j in range(10):
            mean_df.iloc[i, j] = test_score_mean[i, j]
            std_df.iloc[i, j] = test_score_std[i, j]
    if task in ["ideal", "cold user", "ablation", "IPS"]:        
    # 保存
        if not os.path.exists(f"TestRecord/{task}/power={power}/{model_name}"):
            os.makedirs(f"TestRecord/{task}/power={power}/{model_name}")
        params = {
            'log_loss':np.array(test_log_loss).mean(),
            'power':power,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"TestRecord/{task}/power={power}/{model_name}")
        idx = len(files) + 1
        save_path = f"TestRecord/{task}/power={power}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)
        mean_df.to_csv(f"TestRecord/{task}/power={power}/{model_name}/mean.csv")
        std_df.to_csv(f"TestRecord/{task}/power={power}/{model_name}/std.csv")
        for i in range(repeats):
            test_score_i = pd.DataFrame(test_score[i]).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(10)})
            test_score_i.to_csv(f"TestRecord/{task}/power={power}/{model_name}/score-{i+1}.csv")
    elif task == "varying K":
    # 保存
        K = kwargs["K"]
        metric = kwargs["similarity_metric"]
        similarity_of = urllib.parse.quote(kwargs["similarity_method"], safe='')
        if not os.path.exists(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}"):
            os.makedirs(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}")
        params = {
            'log_loss':np.array(test_log_loss).mean(),
            'power':power,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}")
        idx = len(files) + 1
        save_path = f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)
        mean_df.to_csv(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}/mean.csv")
        std_df.to_csv(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}/std.csv")
        for i in range(repeats):
            test_score_i = pd.DataFrame(test_score[i]).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(10)})
            test_score_i.to_csv(f"TestRecord/{task}-{metric}-{similarity_of}/power={power}/K={K}/{model_name}/score-{i+1}.csv")
    elif task == "theta hat":
    # 保存
        n = kwargs["n"]
        if not os.path.exists(f"TestRecord/{task}/power={power}/n={n}/{model_name}"):
            os.makedirs(f"TestRecord/{task}/power={power}/n={n}/{model_name}")
        params = {
            'log_loss':np.array(test_log_loss).mean(),
            'n':n,
            'model name':model_name}
        params.update(kwargs)
        files = os.listdir(f"TestRecord/{task}/power={power}/n={n}/{model_name}")
        idx = len(files) + 1
        save_path = f"TestRecord/{task}/power={power}/n={n}/{model_name}/params-{idx}.json"
        with open(save_path, "w") as f:
            json.dump(params, f)
        mean_df.to_csv(f"TestRecord/{task}/power={power}/n={n}/{model_name}/mean.csv")
        std_df.to_csv(f"TestRecord/{task}/power={power}/n={n}/{model_name}/std.csv")
        for i in range(repeats):
            test_score_i = pd.DataFrame(test_score[i]).rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"}, columns={i:(i+1) for i in range(10)})
            test_score_i.to_csv(f"TestRecord/{task}/power={power}/n={n}/{model_name}/score-{i+1}.csv")
    

    

    
    
    
    




