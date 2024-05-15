import codecs
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.cluster import KMeans


def load_movielens(addr):
    with codecs.open(addr, "r", "utf-8", errors="ignore") as f:
        data = pd.read_csv(f, delimiter='\t', header=None).loc[:, :2]
        data.rename(columns={0:'user', 1:'item', 2:'rate'}, inplace=True)
    data.user, data.item = data.user-1, data.item-1
    num_users, num_items = data.iloc[:, 0].max()+1, data.iloc[:, 1].max()+1
    return data, num_users, num_items


def generate_click_data(data, num_users, num_items, power, user_min_click=0, item_min_click=0,  eps=1e-4):
    """
    eps : 对IPS进行最小切割
    user_min_click : 保留的user的最小点击数
    item_min_click : 保留的item的最小点击数
    返回：列为User, Item, r, theta, R, O, Y, IPS(clip)；num_users, num_items
    """
    r = data[:, 2].view(num_users, num_items)
    theta = torch.sigmoid(torch.logit(data[:, 3].view(num_users, num_items))*power)
    R = torch.bernoulli(r) 
    O = torch.bernoulli(theta)
    Y = R*O
    u_indice = torch.sum(Y, axis=1) >= user_min_click
    i_indice = torch.sum(Y, axis=0) >= item_min_click
    num_users = torch.sum(u_indice)
    num_items = torch.sum(i_indice)
    r = r[u_indice, :][:, i_indice]
    theta = theta[u_indice, :][:, i_indice]
    R = R[u_indice, :][:, i_indice]
    O = O[u_indice, :][:, i_indice]
    Y = Y[u_indice, :][:, i_indice]
    IPS = torch.sum(Y, axis=0)/torch.max(torch.sum(Y, axis=0))
    IPS = torch.sqrt(IPS).repeat(num_users, 1)
    IPS = torch.clip(IPS, eps, 1)
    
    return torch.tensor(np.c_[pd.DataFrame(r).stack().reset_index().values,
                              pd.DataFrame(theta).stack().values,
                              pd.DataFrame(R).stack().values,
                              pd.DataFrame(O).stack().values,
                              pd.DataFrame(Y).stack().values,
                              pd.DataFrame(IPS).stack().values]
                              ),num_users, num_items


def get_train_data_with_ideal_theta(click_data, model_name, **kwargs):
    """
    model_name : 
    1. True
    2. Naive
    3. REL-MF-CLIP
    4. ERM-MF-PRIOR
    5. ERM-MF-ESTIMATE
    
    user | item | r | theta | R | O | Y | IPS | (r_hat) 
                                                            -> ...| label
    """
    if model_name == "True":
        label = click_data[:, 4].reshape(-1, 1)
    elif model_name == "Naive":
        label = click_data[:, 6].reshape(-1, 1)
    elif model_name == "REL-MF-CLIP":
        label = (click_data[:, 6]/torch.clip(click_data[:, 3], kwargs["IPS_clip"], 1)).reshape(-1, 1)
    elif model_name == "ERM-MF-PRIOR":
        r_hat = get_r_hat(click_data, 6, 3, kwargs["K"] if "K" in kwargs.keys() else 1,
                         kwargs["similarity_metric"] if "similarity_metric" in kwargs.keys() else "cosine",
                         kwargs["similarity_method"] if "similarity_method" in kwargs.keys() else "relevance")
        tau2 = 0
        sigma2 = r_hat*(1-r_hat)/(1+kwargs['phi_r'])    
        lam = kwargs["lam"]
        label = get_ERM_label(click_data[:, 6], r_hat, click_data[:, 3], lam, sigma2, tau2).reshape(-1, 1)
    elif model_name == "ERM-MF-ESTIMATE":
        tau2 = 0
        r_hat = click_data[:, -1]
        sigma2 = r_hat*(1-r_hat)/(1+kwargs['phi_r'])
        lam = kwargs["lam"]
        label = get_ERM_label(click_data[:, 6], r_hat, click_data[:, 3], lam, sigma2, tau2).reshape(-1, 1)
    data = torch.hstack([click_data, label])
    return data


def get_train_data_with_theta_hat(click_data, model_name, **kwargs):
    """
    click_data : user | item | r | theta | R | O | Y | [IPS | theta_hat] | (r_hat)
    实验2的函数
    model_type : 
    3. REL-MF-CLIP
    4. ERM-MF-PRIOR
    5. ERM-MF-ESTIMATE
    """
    if model_name == "REL-MF-CLIP":
        label = (click_data[:, 6]/torch.clip(click_data[:, -1], kwargs["IPS_clip"], 1)).reshape(-1, 1)
    elif model_name == "ERM-MF-PRIOR":
        click_data = click_data.clone()
        click_data[:, -1] = torch.clip(click_data[:, -1] + kwargs["mu"], 1e-4, 1)
        r_hat = get_r_hat(click_data, 6, -1, kwargs["K"] if "K" in kwargs.keys() else 1,
                         kwargs["similarity_metric"] if "similarity_metric" in kwargs.keys() else "cosine",
                          kwargs["similarity_method"] if "similarity_method" in kwargs.keys() else "relevance")
        tau2 = click_data[:, -1]*(1-click_data[:, -1])/(1+kwargs["phi_theta"])
        sigma2 = r_hat*(1-r_hat)/(1+kwargs['phi_r'])
        lam = kwargs["lam"]
        label = get_ERM_label(click_data[:, 6], r_hat, click_data[:, -1], lam, sigma2, tau2).reshape(-1, 1)
    elif model_name == "ERM-MF-ESTIMATE":
        click_data = click_data.clone()
        click_data[:, -2] = torch.clip(click_data[:, -2] + kwargs["mu"], 1e-4, 1)
        tau2 = click_data[:, -2]*(1-click_data[:, -2])/(1+kwargs["phi_theta"])
        r_hat = click_data[:, -1]
        sigma2 = r_hat*(1-r_hat)/(1+kwargs['phi_r'])
        lam = kwargs["lam"]
        label = get_ERM_label(click_data[:, 6], r_hat, click_data[:, -2], lam, sigma2, tau2).reshape(-1, 1)
    data = torch.hstack([click_data, label])
    return data


#def get_similarity_hat(data):
#    D = torch.clip(data.sum(axis=0).reshape([-1, 1]) + data.sum(axis=0) - 2*(data.T).matmul(data), 0)
#    D.diag = 0
#    D = torch.sqrt(D)
#    return D

def get_similarity_hat(data):
    D = torch.clip(data.sum(axis=0).reshape([-1, 1]) + data.sum(axis=0) - 2*(data.T).matmul(data), 0)
    return torch.sqrt(D)

def get_r_hat(data, Y_col, theta_col, K=1, similarity_metric="cosine", similarity_method="relevance"):
    Y = data[:, Y_col].reshape([943, 1682])
    theta = data[:, theta_col].reshape([943, 1682])
    r_hat = torch.zeros([943, 1682])
    if K > 1 and K <= 1682:
        if similarity_method == "relevance":
            R = data[:, 4].reshape([943, 1682])
            similarity = pairwise_distances(R.T, metric=similarity_metric)
        elif similarity_method == "click":
            R = data[:, 6].reshape([943, 1682])
            similarity = pairwise_distances(R.T, metric=similarity_metric)
        elif similarity_method == "relevance-hat":
            similarity = get_similarity_hat(Y/torch.clip(theta, 0))
        else:
            similarity = get_similarity_hat(torch.tensor(np.load("data/r_hat.npy").reshape([943, 1682])))
        kmeans = KMeans(n_clusters=K, init='random', n_init=10, random_state=0)
        kmeans.fit(similarity)
        labels = kmeans.labels_
    elif K > 1682:
        r_hat = 0.5*torch.ones([943, 1682])
        return r_hat.flatten()
    else:
        labels = np.zeros(1682)
    for user in tqdm(range(943)):
        for k in range(K):
            Y_user_k = Y[user, labels==k]
            theta_user_k = theta[user, labels==k]
            r_hat[user, labels==k] = get_r_hat_for_one_user(Y_user_k, theta_user_k)
    return r_hat.flatten()
        
def get_r_hat_for_one_user(Y, theta, tol=1e-4):
    if Y.sum() == 0:
        return 0.5
    def equation(a):
        return (Y/a).sum()-((1-Y)*theta/(1-a*theta)).sum()
    upper = 1
    lower = 0
    while upper-lower>tol:
        mid = (upper+lower)/2
        value = equation(mid)
        if abs(value) < tol:
            break
        elif value > 0:
            lower = mid
        else:
            upper = mid
    return mid

def get_ERM_label(Y, r_hat, theta_hat, lam, sigma2, tau2):
    f1 = theta_hat*sigma2/(lam + sigma2*theta_hat**2 + tau2*r_hat**2 + tau2*sigma2)
    f0 = r_hat - sigma2*theta_hat**2*r_hat/(lam + sigma2*theta_hat**2 + tau2*r_hat**2 + tau2*sigma2)
    label = f1*Y + f0
    return label


def get_data_for_computing_score(test_data, num_pos=1, num_neg=100):
    users = test_data[test_data[:, 4]==1, 0].to(dtype=torch.int) # test中R = 1的user
    data_for_scoring = {}
    for user in set(users.detach().cpu().numpy()):
        user_data = test_data[test_data[:, 0]==user]
        # 随机抽取正样例
        pos_data = user_data[user_data[:, 4]==1]
        pos_data_for_scoring = pos_data[torch.randperm(len(pos_data))[:num_pos]]
        # 随机抽取负样例
        neg_data = user_data[user_data[:, 4]==0]
        neg_data_for_scoring = neg_data[torch.randperm(len(neg_data))[:num_neg]]
        data_for_scoring[user] = torch.vstack([pos_data_for_scoring, neg_data_for_scoring])
    return data_for_scoring

    
def get_theta_hat(data, n):
    """
    data : user | item | r | theta | R | O | Y | IPS
    """
    O = torch.bernoulli(data[:, 3].repeat([n, 1]))
    theta_hat = torch.clip(O.mean(axis=0).reshape([-1, 1]), 1e-4, 1)
    data = torch.hstack([data, theta_hat])
    return data

def get_cold_start_users(click_data, user_min_click):
    users = set(click_data[:, 0].detach().cpu().numpy())
    cold_start_users = []
    for user in users:
        Y_user = click_data[click_data[:, 0]==user, 6]
        if Y_user.sum() <= user_min_click:
            cold_start_users.append(user)
    return set(cold_start_users)

def get_rare_items(click_data, item_min_click):
    items = set(click_data[:, 1].detach().cpu().numpy())
    rare_items = []
    for item in items:
        Y_item = click_data[click_data[:, 1]==item, 6]
        if Y_item.sum() <= item_min_click:
            rare_items.append(item)
    return set(rare_items)
         
    
def get_data_for_cold_start_testing(train, test, user_min_click=6):
    cold_users = get_cold_start_users(train, user_min_click)
    cold_users = list(set(test[:, 0].detach().cpu().numpy())&cold_users)
    cold_data = test[(test[:, 0].unsqueeze(1) == torch.tensor(cold_users).unsqueeze(0).repeat([len(test), 1])).any(dim=1)]
    return cold_data
        