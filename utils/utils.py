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

colors = ["red", "blue", "green", "black", "orange", "purple"]

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

def read_scores(save_path):
    Data = {}
    for file1 in os.listdir(save_path):
        if len(re.findall(r'\d+\.\d+|\d+', file1)) == 0:
            continue
        p = float(re.findall(r'\d+\.\d+|\d+', file1)[0])
        path1 = os.path.join(save_path, file1)
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
    task : 
    1. ideal comparison
    2. ideal cold users
    3. ablation experiment
    4. impact of theta
    5. IPS estimation
    
    Data = {
        power / percentage : [{
            "model_type" : model_type,
            "scores" : {
                metric : {
                    {k : score}
                }
            }
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
    elif task == 2:
        for power in [1.0, 1.5, 2.0]:
            for metric in ["RECALL", "MAP", "NDCG"]:
                for data in Data[power]:
                    scores = data["scores"][metric].values()
                    k =  data["scores"][metric].keys()
                    plt.plot(k, scores, label=data["model_type"], color=colors[data["model_type"]])
                    plt.title(f"cold user experiment of power={power}")
                    plt.xlabel("k")
                    plt.ylabel(f"{metric}")
                    plt.legend()
                if not os.path.exists(f"Images/task={task}/power={power}"):
                    os.makedirs(f"Images/task={task}/power={power}")
                img_path = f"Images/task={task}/power={power}/metric={metric}.png"
                plt.savefig(img_path)
                plt.show()
                plt.clf()
    elif task == 3:
        for power in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for metric in ["RECALL", "MAP", "NDCG"]:
                for data in Data[power]:
                    scores = data["scores"][metric].values()
                    k =  data["scores"][metric].keys()
                    plt.plot(k, scores, label=data["model_type"], color=colors[data["model_type"]])
                    plt.title(f"ablation experiment of power={power}")
                    plt.xlabel("k")
                    plt.ylabel(f"{metric}")
                    plt.legend()
                if not os.path.exists(f"Images/task={task}/power={power}"):
                    os.makedirs(f"Images/task={task}/power={power}")
                img_path = f"Images/task={task}/power={power}/metric={metric}.png"
                plt.savefig(img_path)
                plt.show()
                plt.clf()    
    elif task == 4:
        for power in [1, 0.75, 0.5, 0.25]:
            for metric in ["RECALL", "MAP", "NDCG"]:
                for data in Data[power]:
                    scores = data["scores"][metric].values()
                    k =  data["scores"][metric].keys()
                    plt.plot(k, scores, label=data["model_type"], color=colors[data["model_type"]])
                    plt.title(f"Theta experiment of power={power}")
                    plt.xlabel("k")
                    plt.ylabel(f"{metric}")
                    plt.legend()
                if not os.path.exists(f"Images/task={task}/power={power}"):
                    os.makedirs(f"Images/task={task}/power={power}")
                img_path = f"Images/task={task}/power={power}/metric={metric}.png"
                plt.savefig(img_path)
                plt.show()
                plt.clf()
    elif task == 5:
        for power in [1]:
            for metric in ["RECALL", "MAP", "NDCG"]:
                for data in Data[power]:
                    scores = data["scores"][metric].values()
                    k =  data["scores"][metric].keys()
                    plt.plot(k, scores, label=data["model_type"], color=colors[data["model_type"]])
                    plt.title(f"IPS experiment of power={power}")
                    plt.xlabel("k")
                    plt.ylabel(f"{metric}")
                    plt.legend()
                if not os.path.exists(f"Images/task={task}/power={power}"):
                    os.makedirs(f"Images/task={task}/power={power}")
                img_path = f"Images/task={task}/power={power}/metric={metric}.png"
                plt.savefig(img_path)
                plt.show()
                plt.clf()