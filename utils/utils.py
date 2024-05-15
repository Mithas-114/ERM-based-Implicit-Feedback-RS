import os
import re
import codecs
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

colors = {"True":"red", "Naive":"blue", "REL-MF-CLIP":"green",  "ERM-MF-PRIOR":"orange", "ERM-MF-ESTIMATE":"purple"}

def seed_everything(seed=42):
    """
    set all the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_all(model, model_type, power, repeat, train_loss_list, val_loss_list, test_loss_list):
    """
    save all the embeddings and curves
    """
    model_path = f"Simulate/{model_type}/power={power}-{repeat}.pth"
    torch.save(model.state_dict(), model_path)
    img_path = f"Simulate/{model_type}/power={power}-{repeat}.jpg"
    plt.plot(train_loss_list, label="train loss")
    plt.plot(val_loss_list, label="val loss")
    plt.plot(test_loss_list, label="test loss")
    plt.title(f"train-val-test loss of power={power} and repeat={repeat}")
    plt.xlabel("epochs")
    plt.ylabel("log loss")
    plt.legend()
    plt.savefig(img_path)
    plt.clf() 
    
    
def search_eps_for_rating(r):
    expectation = 0.52*torch.sigmoid(torch.tensor(-2)) + 0.23*torch.sigmoid(torch.tensor(-1))\
                  + 0.15*torch.sigmoid(torch.tensor(0)) + 0.07*torch.sigmoid(torch.tensor(1)) + 0.03*torch.sigmoid(torch.tensor(2))
    def equation(eps):
        return torch.mean(torch.sigmoid(r-eps))-expectation
    upper = 5
    lower = 0
    while upper-lower > 1e-6:
        mid = (upper+lower)/2
        if abs(equation(mid)) < 1e-6:
            break
        elif equation(mid) > 0:
            lower = mid
        else:
            upper = mid
    return mid

def read_scores(save_path):
    Data = {}
    for file1 in os.listdir(save_path):
        Data[p] = []

def read_scores(save_path):
    Data = {}
    for task_file in os.listdir(save_path):
        Data["task"] = task
        Data["data"] = []
        for num_file1 in os.listdir(os.path.join(save_path, task_file)):
            data = {}
            p = int(re.findall(r'\b\d+(\.\d+)?\b', num_file1)[0])
            data["num1"] = p
            data["data"] = []
            for file2 in os.listdir(os.path.join(save_path, task_file, num_file1)):
                data_ = {}
                q = re.findall(r'\b\d+(\.\d+)?\b', num_file1)
                if len(q) == 0:
                    data_["model name"] = file2
                    data_["scores"] = {}
                    scores = pd.read_csv(os.path.join(save_path, num_file1, file2, "mean.csv"))\
                                .iloc[:, 1:].rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"})
                    for metric in ["NDCG", "MAP", "RECALL"]:
                        data_["scores"][metric] = dict(data.loc[metric, ])
                else:
                    q = int(q[0])
                    data_["num2"] = q
                    data_["data"] = []
                    for file3 in os.listdir(os.path.join(save_path, task_file, num_file1, file2)):
                        data__ = {}
                        data__["model name"] = file3
                        data__["scores"] = {}
                        scores = pd.read_csv(os.path.join(save_path, num_file1, file2, file3, "mean.csv"))\
                                        .iloc[:, 1:].rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"})
                        for metric in ["NDCG", "MAP", "RECALL"]:
                            data__["scores"][metric] = dict(data.loc[metric, ])
                        data_["data"].append(data__)
                data["data"].append(data_)
            Data["data"].append(data)
        return Data
                
                
            
        

        Data[p] = []
        for file2 in os.listdir(path1):
            if len(re.findall(r'\d+', file2)) == 0:
                continue
            Datap = {}
            t = int(re.findall(r'\d+', file2)[0])
            Datap["model_type"] = t
            Datap["scores"] = {}
            data = pd.read_csv(os.path.join(path1, file2, "mean.csv")).iloc[:, 1:].rename(index={0:"NDCG", 1:"MAP", 2:"RECALL"})
            for metric in ["NDCG", "MAP", "RECALL"]:
                Datap["scores"][metric] = dict(data.loc[metric, ])
            Data[p].append(Datap)
    return Data
    

def plot(task, Data):
    """  
    Data = {
        'task' : task
        'data' : [{
            'num':num,
            'data' : [{
                'model name':model_name,
                'data':data
            }]
        }]
    }
    """
    if task == 1:
        for power in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for metric in ["RECALL", "MAP", "NDCG"]:
                for data in Data[power]:
                    scores = data["scores"][metric].values()
                    k =  data["scores"][metric].keys()
                    plt.plot(k, scores, label=data["model_type"], color=colors[data["model_type"]])
                    plt.title(f"ideal experiment of power={power}")
                    plt.xlabel("k")
                    plt.ylabel(f"{metric}")
                    plt.legend()
                if not os.path.exists(f"Images/task={task}/power={power}"):
                    os.makedirs(f"Images/task={task}/power={power}")
                img_path = f"Images/task={task}/power={power}/metric={metric}.png"
                plt.savefig(img_path)
                plt.show()
                plt.clf()