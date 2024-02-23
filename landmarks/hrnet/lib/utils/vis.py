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
import torchvision
import cv2
import os
import pdb
from core.inference import get_max_preds
from pathlib import Path


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    #pdb.set_trace()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    
    imagename=file_name.split('.')[0]
    #cv2.imwrite(imagename+'.png', ndarr)
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 1, [255, 0, 0], 2)
            k = k + 1
            
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)
    
    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        
        
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def write(path, pts, ver='version:1'):
    lines = [ver]
    lines.append('n_points:{}'.format(len(pts)))
    lines.append("{")
    for (x,y) in pts:
        lines.append('{} {}'.format(x,y))
    lines.append("}")
    with open(path,'w') as f:
        f.write('\n'.join(lines))
    return 



def save_batch_joints_image_pts(meta, input, joints_pred,outputdir):
    #保存生成的pts
    #input ---- N 3 H W 
    #joints ---- N C  2  预测关键点坐标 ，可以通过output求得  坐标已扩大网络层的倍数了
    
    filename=''
    ptsname=''
    
    
    #此处需要归一化，不归一化图像不正常
    batch_image = input.clone()
    min = float(batch_image.min())
    max = float(batch_image.max())
    batch_image.add_(-min).div_(max - min + 1e-5)
    
    #遍历每一个batch
    for i in range(batch_image.shape[0]):   
        point_list=[]
        image = batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()      
        filename=meta['image'][i] 
        #imagename=filename.split('/')[-1]
        
        name_crop = filename.split('.')
        com_name= name_crop[0].split('/')[3]
        ptsname=com_name+'.pts'
        imagename=com_name+'.png'
        
        
        #遍历每一个关键点
        for idx in range(joints_pred.shape[1]):
            point = joints_pred[i,idx,:]
            point_list.append(point.tolist())
          
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        
        ptspath=os.path.join(outputdir, ptsname)
        imgpath=os.path.join(outputdir, imagename)
        
        write(ptspath,point_list)
        cv2.imwrite(imgpath, image)




def save_batch_final_joints_image_pts(meta, final_joints_pred, outputdir):
#保存生成的pts
    #input ---- meta[image]
    #final_joints_pred ---- N C  2  经过仿射变换之后的逆矩阵之后的点
    
    #pdb.set_trace()
    filename=''
    ptsname=''   
    
    
    #遍历每一个batch
    for i in range(final_joints_pred.shape[0]):   
        point_list=[]
        filename=meta['image'][i] 
        
        #读入图片
        data_numpy = cv2.imread(
                filename, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)    
        #data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        
        #imagename=filename.split('/')[-1]
        
        name_crop = filename.split('.')
        com_name= name_crop[0].split('/')[4]
        ptsname = com_name+'.pts'
        
        #不改变图片名
        imagename = filename.split('/')[-1]   
        
        
        #遍历每一个关键点
        for idx in range(final_joints_pred.shape[1]):
            point = final_joints_pred[i,idx,:]
            point_list.append(point.tolist())
                              
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        ptspath=os.path.join(outputdir, ptsname)
        imgpath=os.path.join(outputdir, imagename)
        
        write(ptspath,point_list)
        cv2.imwrite(imgpath, data_numpy)



def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return
    
    #保存原图的gt
    if config.DEBUG.SAVE_BATCH_IMAGES_GT: 
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    #保存gt的热图
    if config.DEBUG.SAVE_HEATMAPS_GT:     
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
        
        
    #保存预测出的热图
    if config.DEBUG.SAVE_HEATMAPS_PRED:   
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

