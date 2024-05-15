import torch
import numpy as np

"""
true_score : 真实分数。在验证时是Y，测试时是R
pred_score : 预测分数
k : 排序位置
IPS ：如果是none，说明是test；否则是val

验证时，使用SNIPS度量进行参数选择。
测试时，使用标准度量进行评价。
"""

def cal_DCG(true_score, pred_score, k, IPS=None):

    if IPS is None:
        # test
        rel = true_score[pred_score.argsort(descending=True)]
        dcg_score = torch.tensor(0)
        if not torch.sum(rel) == 0:
            dcg_score = torch.sum(rel[:k]/np.log2(np.arange(1,k+1)+1)) 
        return dcg_score
    else:
        # val
        true_score_ = true_score/IPS
        rel = true_score_[pred_score.argsort(descending=True)]
        dcg_score = torch.tensor(0)
        if not torch.sum(rel) == 0:
            dcg_score = torch.sum(rel[:k]/np.log2(np.arange(1,k+1)+1))/torch.sum(rel)
        return dcg_score

def cal_NDCG(true_score, pred_score, k, IPS=None):
    """
    在val时，只有pred_score排序与true_score/IPS排序相同时，NDCG=1. 
    在test时，只有pred_score排序与true_score排序相同时，NDCG=1. 
    """
    if IPS is not None:
        true_score_ = true_score/IPS # val
    else:
        true_score_ = true_score # test
    rel = true_score_[pred_score.argsort(descending=True)]
    true_rel = true_score_[true_score_.argsort(descending=True)]
    ndcg_score = torch.tensor(0)
    if not torch.sum(rel) == 0:
        dcg_score = torch.sum(rel[:k]/np.log2(np.arange(1,k+1)+1))
        idcg_score = torch.sum(true_rel[:k]/np.log2(np.arange(1,k+1)+1))
        ndcg_score = dcg_score/idcg_score
    return ndcg_score

def cal_RECALL(true_score, pred_score, k, IPS=None):

    if IPS is not None:
        true_score_ = true_score/IPS # val
    else:
        true_score_ = true_score # test
    recall_score = torch.tensor(0)
    rel = true_score_[pred_score.argsort(descending=True)]
    if not torch.sum(rel) == 0:
        recall_score = torch.sum(rel[:k])/torch.sum(rel)
    return recall_score
    
def cal_AP(true_score, pred_score, k, IPS=None):
    """
    在val计算MAP时，由于Y/theta可能超过1，因此P可能超过1，使得MAP也有可能超过1. 
    但是test计算MAP不会发生这种情况。
    所以不使用MAP进行参数选择。
    """
    if IPS is not None:
        true_score_ = true_score/IPS # val
    else:
        true_score_ = true_score # test
    rel = true_score_[pred_score.argsort(descending=True)]
    average_precision_score = torch.tensor(0)
    if not torch.sum(rel) == 0:
        rel_np = rel.detach().cpu().numpy().astype(np.float64)
        def cal_P(i):
            return np.mean(rel_np[:i])
        cal_P = np.vectorize(cal_P)
        rel_np_k = rel_np[:k]
        if not np.sum(rel_np_k) == 0:
            average_precision_score = np.sum(cal_P(np.arange(1, k+1)[rel_np_k>0])*rel_np_k[rel_np_k>0])
            average_precision_score /= np.sum(rel_np)
        else:
            average_precision_score = 0
        return torch.tensor(average_precision_score)
    return average_precision_score






