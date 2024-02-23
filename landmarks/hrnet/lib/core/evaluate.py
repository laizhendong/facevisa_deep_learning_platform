# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds

import pdb  
from math import sqrt

def calc_dists(preds, target, normalize):
    #pdb.set_trace()
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):                                                               #遍历每个batch
        for c in range(preds.shape[1]):                                                           #遍历每个关键点
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:                                       #判断生成的target坐标是否大于1，常规都会大于1
                normed_preds = preds[n, c, :] / normalize[n]                                      #取每一个batch的预测
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)                       #二范数      [关键点数，batchsize]
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)                                                             #返回值是（batchsize，）里面全是True ，batchsize个True
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal                            #如果里面的值存在不等于-1的就给返回true,将距离值转换成百分比  对2个点坐标小于0.1的进行求和
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''

    #此处的output和target均是热图 
    
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)                                                             #返回热图中的关键点坐标索引
        target, _ = get_max_preds(target)
        
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10                                  #这里不知道为啥除以10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):                                                                        #遍历每一个关键点
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def MSE_eval_points_big(preds_points, meta):
    #pdb.set_trace()
    preds = preds_points[:,:].astype(np.float32)
    target = meta['orijoints'].numpy().astype(np.float32)
    dist_rlt=0
    #遍历每一个batch
    for b in range(preds.shape[0]):  
        #一张图片的点
        if len(preds[b]) == len(target[b]):                                                          #判断一下点数据是否相等
            for idx in range(preds.shape[1]):
                p = preds[b,idx,:]
                gt = target[b,idx,0:2]
                
                dist = sqrt(sum(abs(p-gt)**2)) /(1+0.00001)
                dist_rlt+=dist
    return dist_rlt/(len(preds)*preds.shape[1])

    
    
    
def MSE_eval_points_small(preds, meta):
  
    preds = preds.astype(np.float32)
    target = meta['joints'].numpy().astype(np.float32)
    dist_rlt = 0
    for n in range(preds.shape[0]):                                                                  #遍历每个batch                                                               
        for c in range(preds.shape[1]):                                                              #遍历每个关键点
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:                                          #判断生成的target坐标是否大于1，常规都会大于1
                p = preds[n, c, :]                                                                   #取每一个batch的预测
                gt = target[n, c, 0:2]
                #dists = sum ([p-gt])**2 /(len(preds[n])+0.00001)
                dists = sqrt(sum((p-gt)**2))/(1+0.00001)
                dist_rlt+=dists

    return dist_rlt/(len(preds)*preds.shape[1])   
  
    
    
    
    
