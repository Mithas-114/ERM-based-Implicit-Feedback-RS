import os
import torch
import codecs
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def preprocess_yahoor3(addr, save_path, rating_threshold=4, user_threshold=5):
    """
    train : user | item | Y 
    test : user | item | R 
    IPS : IPS score for training set
    """
    # load dataset
    col = {0: 'user', 1: 'item', 2: 'rate'}
    with codecs.open(os.path.join(addr, "train.txt"), "r", "utf-8", errors="ignore") as f:
        data_train = pd.read_csv(f, delimiter="\t", header=None)
        data_train.rename(columns=col, inplace=True)
    with codecs.open(os.path.join(addr, "test.txt"), "r", "utf-8", errors="ignore") as f:
        data_test = pd.read_csv(f, delimiter="\t", header=None)
        data_test.rename(columns=col, inplace=True)
    num_users, num_items = data_train.user.max(), data_train.item.max()
    for _data in [data_train, data_test]:
        _data.user, _data.item = _data.user-1, _data.item-1
        # convert ratings to relevance
        _data.rate[_data.rate<rating_threshold] = 0
        _data.rate[_data.rate>=rating_threshold] = 1
    # construct training data
    train = data_train.values.astype(np.int64)
    train = train[train[:, 2] == 1, :2]
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2].astype(np.int64)
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=np.int64)
    train = np.r_[np.c_[train, np.ones(train.shape[0], dtype=np.int64)], 
                  np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0], dtype=np.int64)]]
    # compute IPS
    _, item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)
    IPS = (item_freq / item_freq.max()) ** 0.5
    # filter users
    users, user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)
    train = train[np.isin(train[:, 0], users[user_freq>=user_threshold])]
    # reset user id
    filtered_users = set(train[:, 0])
    id_map = {old:new for old, new in zip(filtered_users, list(range(len(filtered_users))))}
    id_map = np.vectorize(id_map.get)
    train[:, 0] = id_map(train[:, 0])
    # construct testing data
    test = data_test.values.astype(np.int64)
    test = test[np.isin(test[:, 0], users[user_freq>=user_threshold])]
    test[:, 0] = id_map(test[:, 0])
    # save datasets
    train = train[np.lexsort((train[:, 1], train[:, 0]))]
    test = test[np.lexsort((test[:, 1], test[:, 0]))]
    np.save(os.path.join(save_path, "train.npy"), arr=train)
    np.save(os.path.join(save_path, "test.npy"), arr=test)
    np.save(os.path.join(save_path, "IPS.npy"), arr=IPS)
    np.save(os.path.join(save_path, "item_freq.npy"), arr=item_freq)

def preprocess_coat(addr, save_path=None, rating_threshold=4, user_threshold=5):
    # 读取train
    with open(os.path.join(addr, "train.ascii"), 'r') as file:
        file.seek(0)
        lines = file.readlines()
        train = []
        for line in lines:
            line = [int(rating) for rating in line.strip('\n').split(' ')]
            train.append(np.array(line))
    train = pd.DataFrame(np.vstack([line for line in train])).stack().reset_index().values
    train_ = train.copy()
    train_[train[:, 2] >= rating_threshold, 2] = 1
    train_[train[:, 2] < rating_threshold, 2] = 0
    # 计算IPS
    Y = train_[:, -1].reshape([290, 300])
    item_freq = Y.sum(axis=0)
    IPS = np.clip((item_freq / item_freq.max()) ** 0.5, 0.01, 1)
    # 处理train
    users, user_freq = np.unique(train_[train_[:, 2] == 1, 0], return_counts=True)
    train = train_[np.isin(train_[:, 0], users[user_freq>=user_threshold])]
    filtered_users = set(train[:, 0])
    id_map = {old:new for old, new in zip(filtered_users, list(range(len(filtered_users))))}
    id_map = np.vectorize(id_map.get)
    train[:, 0] = id_map(train[:, 0])
    
    with open(os.path.join(addr, "test.ascii"), 'r') as file:
        file.seek(0)
        lines = file.readlines()
        test = []
        for line in lines:
            line = [int(rating) for rating in line.strip('\n').split(' ')]
            test.append(np.array(line))
        test = pd.DataFrame(np.vstack([line for line in test])).stack().reset_index().values
        test = test[test[:, 2]>0]
        test_ = test.copy()
        test_[test[:, 2] >= rating_threshold, 2] = 1
        test_[test[:, 2] < rating_threshold, 2] = 0
    test = test_[np.isin(test[:, 0], users[user_freq>=user_threshold])]
    test[:, 0] = id_map(test[:, 0])
    train = train[np.lexsort((train[:, 1], train[:, 0]))]
    test = test[np.lexsort((test[:, 1], test[:, 0]))]
    
    np.save(os.path.join(save_path, "train.npy"), arr=train)
    np.save(os.path.join(save_path, "test.npy"), arr=test)
    np.save(os.path.join(save_path, "IPS.npy"), arr=IPS)
    np.save(os.path.join(save_path, "item_freq.npy"), arr=item_freq)
        
    
def get_train_data(addr, model_name, **kwargs):
    """
    model_name : 
    1. "WMF"
    2. "REL-MF-CLIP"
    3. "ERM-MF-PRIOR"
    4. "ERM-MF-ESTIMATE"
    """
    train = np.load(os.path.join(addr, "train.npy"))
    IPS = np.load(os.path.join(addr, "IPS.npy"))
    IPS_ = np.tile(IPS, (int(len(train)/len(IPS)), 1)).flatten()
    if model_name == "WMF":
        """
        train : user | item | Y(label) | IPS
        """
        train = np.c_[train, IPS_]
    elif model_name == "REL-MF-CLIP" :
        """
        train : user | item | Y | IPS | label
        """
        IPS_ = np.clip(IPS_, kwargs["IPS_clip"], 1)
        label = (train[:, 2]/IPS_)
        train = np.c_[train, IPS_, label]
    elif model_name == "ERM-MF-PRIOR":
        """
        train : user | item | Y | IPS | r_hat | label
        """    
        IPS_ = np.clip(IPS_+kwargs["mu"], 1e-2, 1)
        train = np.c_[train, IPS_]
        r_hat = get_r_hat(train, K=kwargs["K"], r_default=kwargs["r_default"])
        sigma2 = r_hat*(1-r_hat)/(1+kwargs["phi_r"])
        tau2 = IPS_*(1-IPS_)/(1+kwargs["phi_theta"])
        lam = kwargs["lam"]
        label = get_ERM_label(train[:, 2], r_hat, IPS_, lam, sigma2, tau2)
        train = np.c_[train, r_hat, label]
    elif model_name == "ERM-MF-ESTIMATE":
        """
        train : user | item | Y | IPS | r_hat | label
        """
        IPS_ = np.clip(IPS_+kwargs["mu"], 1e-2, 1)
        r_hat = np.load(os.path.join(addr, "r_hat.npy"))   
        sigma2 = r_hat*(1-r_hat)/(1+kwargs["phi_r"])
        tau2 = IPS_*(1-IPS_)/(1+kwargs["phi_theta"])
        lam = kwargs["lam"]
        label = get_ERM_label(train[:, 2], r_hat, IPS_, lam, sigma2, tau2)
        train = np.c_[train, IPS_, r_hat, label]
    else:
        raise ValueError()
    train = torch.tensor(train, dtype=torch.float64)
    return train

def get_test_data(addr):
    test = np.load(os.path.join(addr, "test.npy"))
    test = torch.tensor(test, dtype=torch.float64)
    return test

#def get_item_clusters(labels, K=1):
#    S = (labels**2).sum(axis=0)
#    similarity = np.sqrt(S.reshape([-1, 1])+S-2*labels.T.dot(labels))
#    cluster = KMeans(n_clusters=K, random_state=0).fit(similarity)
#    return cluster.labels_

def get_item_clusters(labels, K=1):
    data = labels.T
    cos  = cosine_similarity(data)
    cluster = KMeans(n_clusters=K, random_state=0).fit(cos)
    return cluster.labels_

def get_r_hat(data, K=1, r_default=0.1):
    """
    data : 依照user, item主序排列，且可还原为矩阵
    for ERM-MF-PRIOR.
    data : user | item | Y | IPS
    K : 聚类数
    """
    num_items = len(set(data[:, 1]))
    num_users = len(set(data[:, 0]))
    labels = data[:, 2].reshape([num_users, num_items])
    IPS = data[:, 3].reshape([num_users, num_items])
    r_hat = np.zeros([num_users, num_items])
    if K > 1:
        # 聚类
        item_clusters = get_item_clusters(labels, K)
    else:
        item_clusters = np.zeros(num_items)
    for user in tqdm(np.arange(num_users)):
        for i in range(K):
            Y_user = labels[user, item_clusters==i]
            theta_user = IPS[user, item_clusters==i]
            r_hat[user, item_clusters==i] = get_r_hat_for_one_user(Y_user, theta_user, r_default)
    return r_hat.reshape(-1, 1).flatten()

def get_r_hat_for_one_user(Y, theta, tol=1e-4, r_default=0.1):
    if Y.sum() == 0:
        return r_default
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


def split_train_and_val(data, num_users, num_items, need_val_out=False, num_pos=1, num_neg=100):
    """
    leave one out.
    为每个user，随机筛选1个正样例和100个负样例。
    data : user | item | Y | ... 
    need_val_out : 若True，则返回val in和val out. 
    返回train，val in, (val out)
    """
    if num_items < 800:
        num_neg = round(num_items/10)
    data_ = {user:data[user*num_items:(user+1)*num_items, :] for user in range(num_users)}
    train = []
    val_in = []
    val_out = []
    for user, user_data in data_.items():
        # 随机抽取正样例
        pos_data = user_data[user_data[:, 2]==1]
        pos_random_perm = torch.randperm(len(pos_data))
        val_in_pos_data = pos_data[pos_random_perm[:num_pos]]
        train_pos_data = pos_data[pos_random_perm[num_pos:]]
        if need_val_out:
            val_out_pos_data = train_pos_data[:num_pos]
            train_pos_data = train_pos_data[num_pos:]
        # 随机抽取负样例
        neg_data = user_data[user_data[:, 2]==0]
        neg_random_perm = torch.randperm(len(neg_data))
        val_in_neg_data = neg_data[neg_random_perm[:num_neg]]
        train_neg_data = neg_data[neg_random_perm[num_neg:]]
        if need_val_out:
            val_out_neg_data = train_neg_data[:num_neg]
            train_neg_data = train_neg_data[num_neg:]
        # 处理
        val_in.append(torch.vstack([val_in_pos_data, val_in_neg_data]))
        train.append(torch.vstack([train_pos_data, train_neg_data]))
        if need_val_out:
            val_out.append(torch.vstack([val_out_pos_data, val_out_neg_data]))
    val_in = torch.vstack([entry for entry in val_in])
    train = torch.vstack([entry for entry in train])
    if need_val_out:
        val_out = torch.vstack([entry for entry in val_out])
        return train, val_in, val_out
    return train, val_in

def get_cold_user(train, test, num_users, num_items, threshold=6):
    train_ = train[:, 2].reshape([num_users, num_items])
    cold_users = torch.arange(num_users)[train_.sum(axis=1) <= threshold].numpy().astype(np.int32)
    return test[np.isin(test[:, 0].numpy().astype(np.int32), cold_users)]