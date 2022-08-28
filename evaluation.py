# coding=utf-8


import torch
import numpy as np

import util
from generic_utils import Progbar


def l2norm(X):
    """L2-normalize columns of X
    use numpy.array
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / (norm + 1e-10) # avoid divide by ZERO


@util.timer
def hist_sim(im, s, device):
    im = torch.Tensor(im).to(device)
    s = torch.Tensor(s).to(device)
    score = torch.zeros((im.size(0), s.size(0))).to(device)

    im_bs = im.size(0)
    s_bs = s.size(0)
    im = im.unsqueeze(1).expand(-1,s_bs,-1)
    s = s.unsqueeze(0).expand(im_bs,-1,-1)

    for index in range(im.shape[0]):
        im1 = im[index,:,:]
        s1 = s[index,:,:]
        intersection = torch.min(im1, s1).sum(-1)
        union = torch.max(im1, s1).sum(-1)
        score[index, :] = (intersection / union)

    # intersection = torch.min(im,s).sum(-1)
    # union = torch.max(im,s).sum(-1)
    # score = intersection / union
    # print(score.size())
    return score.cpu().numpy()


@util.timer
def cosine_sim(query_embs, retro_embs):
    query_embs = l2norm(query_embs)
    retro_embs = l2norm(retro_embs)

    return query_embs.dot(retro_embs.T)
    # return consine_sim1(query_embs, retro_embs)


def compute_sim(query_embs, retro_embs, measure='cosine', device=torch.device('cpu')):
    if measure == 'cosine':
        return cosine_sim(query_embs, retro_embs)
    elif measure == 'hist':
        return hist_sim(query_embs, retro_embs, device)
    elif measure == 'euclidean':
        raise Exception('Not implemented')
    else:
        raise Exception('%s is invalid' % measure)


def eval_qry2retro(qry2retro_sim, n_qry=1):
    """
    Query->Retrieval
    qry2retro_sim: (n_qry*N, N) matrix of query to video similarity
    """

    assert qry2retro_sim.shape[0] / qry2retro_sim.shape[1] == n_qry, qry2retro_sim.shape
    ranks = np.zeros(qry2retro_sim.shape[0])

    inds = np.argsort(qry2retro_sim, axis=1)

    for index in range(len(ranks)):
        ind = inds[index][::-1]

        rank = np.where(ind == index/n_qry)[0][0]
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mir = (1.0/(ranks+1)).mean()

    return (r1, r5, r10, medr, meanr, mir)

def eval(label_matrix):
    label_matrix = label_matrix.astype(int)
    ranks = np.zeros(label_matrix.shape[0])
    aps = np.zeros(label_matrix.shape[0])
    r1s = np.zeros(label_matrix.shape[0])
    r5s = np.zeros(label_matrix.shape[0])
    r10s= np.zeros(label_matrix.shape[0])

    for index in range(len(ranks)):
        rank = np.where(label_matrix[index]==1)[0] + 1
        if len(rank)==0:
            r1s[index], r5s[index], r10s[index],aps[index],ranks[index]=0,0,0,1/6,6
            continue
        ranks[index] = rank[0]
        aps[index] = np.mean([(i+1.)/rank[i] for i in range(len(rank))])
        r1s[index] = 100.0 * np.mean([rank[0] <= 1])
        r5s[index] = 100.0 * np.mean([rank[0] <= 5])
        r10s[index] = 100.0 * np.mean([rank[0] <= 10])
    #r1, r5, r10 = [100.0*np.mean([x <= k for x in ranks]) for k in [1, 5, 10]]
    r1, r5, r10=r1s.mean(),r5s.mean(),r10s.mean()
    medr=np.floor(np.median(ranks))
    meanr = ranks.mean()
    mir = (1.0/ranks).mean()
    mAP = aps.mean()

    return (r1, r5, r10, medr, meanr, mir, mAP,aps,r1s,r5s,r10s,ranks)

def eval_multigt(label_matrix):
    label_matrix = label_matrix.astype(int)
    meanranks = np.zeros(label_matrix.shape[0])
    midranks=np.zeros(label_matrix.shape[0])
    aps = np.zeros(label_matrix.shape[0])
    r1 = np.zeros(label_matrix.shape[0])
    r5 = np.zeros(label_matrix.shape[0])
    r10 = np.zeros(label_matrix.shape[0])
    for index in range(label_matrix.shape[0]):
        rank = np.where(label_matrix[index]==1)[0] + 1
        if len(rank)==0:
            r1[index], r5[index], r10[index],rank,meanranks[index],midranks[index]=0,0,0,6,6,6

        rank=min(rank)
        meanranks[index]=rank
        midranks[index]=rank
        #res= [100.0 * np.mean([x <= len(rank) for x in rank[:k]]) for k in [1, 5, 10]]
        res = [100.0 * np.mean([ rank[0]<=k]) for k in [1, 5, 10]]

        r1[index], r5[index], r10[index]=res
        aps[index] = np.mean([(i+1.)/rank[i] for i in range(len(rank))])

    medr = np.floor(np.median(midranks))
    meanr = meanranks.mean()
    mir = (1.0/meanranks).mean()
    mAP = aps.mean()
    mr1=r1.mean()
    r5=r5.mean()
    r10=r10.mean()

    return (mr1, r5, r10, medr, meanr, mir, mAP,aps,r1)
