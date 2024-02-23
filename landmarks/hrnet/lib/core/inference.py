# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds
import pdb

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''

    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))   #将图片进行reshape操作 【B ,num, H*W】
    idx = np.argmax(heatmaps_reshaped, 2)                                      #获得最大元素在第2维的索引           等同于找出热图中的最大值，后面的构成维度，B个num*（H*W）的矩阵             
    maxvals = np.amax(heatmaps_reshaped, 2)                                    #获得最大元素在第2维的像素值         shape为【B，Num】

    maxvals = maxvals.reshape((batch_size, num_joints, 1))                     #【B num 1】
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)                         #将idx进行复制，后面再替换里面的坐标值，此刻的坐标为输出图的坐标            得到的是一维的坐标中的索引

    preds[:, :, 0] = (preds[:, :, 0]) % width                                  #将H*W的值转换成行和列的值，x(行)：对框取余，y（列）：对宽相除向下取整
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))                   #返回像素值大于0 一个True的mask矩阵
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask            
    return preds, maxvals                                                      #返回的是最大值索引和最大值


def get_final_preds(config, batch_heatmaps, center, scale):                    #【16,20,128,128】
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):                                        #遍历通道数
            for p in range(coords.shape[1]):                                    #遍历每一个关键点
                hm = batch_heatmaps[n][p]                                       #一张单通道的热图
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))                     #将坐标值加上0.5向下取整
                
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:      #判断坐标值（此时的坐标是位于输入图片的坐标）
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25                         #没看懂为啥这样操作     [-0.25, 0, 0.25]---对坐标值进行微调（根据热图相邻元素之间差距，通过符号函数）

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
