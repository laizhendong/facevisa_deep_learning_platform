import os, cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import random
import time
from scipy.integrate import simps
import pdb
from math import sqrt


def get_label(data_name, label_file, task_type=None):

    label_path = os.path.join('data', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new





def get_meanface(meanface_file, num_nb):
    with open(meanface_file) as f:
        meanface = f.readlines()[0]
    

    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]                     #将str2float
    meanface = np.array(meanface).reshape(-1, 2)            
    

    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]                                      #取每一个点【x,y】
        dists = np.sum(np.power(pt-meanface, 2), axis=1)        #按行求和
        indices = np.argsort(dists)                             #返回的是元素值从小到大排序后的索引值的数组
        meanface_indices.append(indices[1:1+num_nb])            #取10个索引，第0个索引是自己本身
    

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]                  #类，key:values   关键点数
    
    
    
    #m表示的邻近关键点的索引（0-29）     
    #i--第几个关键点              
    #j--表示i相关联的邻近关键点的在meanface中的位置
    #此处进行了邻近关键点映射关系的转换：将之前的第i个关键点有10个相邻的邻近关键点索引  ------>在meanface中的第i个关键点第j个位置的邻近关键点索引
    #{【该处的关键点是哪些关键点的邻近关键点，不含本身】 【该处的关键点为meanface中邻近关键点的位置】}
    

    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            m = meanface_indices[i][j]
            meanface_indices_reversed[m][0].append(i)           #与第i个关键点邻近的关键点索引
            meanface_indices_reversed[m][1].append(j)           #与第i个关键点邻近的关键点索引在与之相关连点的位置

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len                                   #max_len  ！= num_nb
    
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*max_len            
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*max_len
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]     
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]      
  
    

    #将数据直接变成1维       邻近关键点的索引，如1是哪些关键点的邻近索引
    #邻近关键点索引是是哪些关键点邻近索引的第几个    如1是2的邻近关键点的第2个索引，即距离第2近的索引
    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]                          
        reverse_index2 += meanface_indices_reversed[i][1]                           

    return meanface_indices, reverse_index1, reverse_index2, max_len



def compute_loss_pip(outputs_map, outputs_local_x, outputs_local_y, outputs_nb_x, outputs_nb_y, labels_map, labels_local_x, labels_local_y, labels_nb_x, labels_nb_y,  criterion_cls, criterion_reg, num_nb):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()   
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    #预测输出偏移量
    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)


    #标签偏移量
    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map, labels_map)                                   #直接用图进行回归    MSE
    loss_x = criterion_reg(outputs_local_x_select, labels_local_x_select)               #转成一维            L1
    loss_y = criterion_reg(outputs_local_y_select, labels_local_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)
    return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y

def train_model(det_head, net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
    for epoch in range(num_epochs):
      
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            
            if det_head == 'pip':
                inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y = data
                inputs = inputs.to(device)
                labels_map = labels_map.to(device)
                labels_x = labels_x.to(device)
                labels_y = labels_y.to(device)
                labels_nb_x = labels_nb_x.to(device)
                labels_nb_y = labels_nb_y.to(device)
                outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)  
                
                #训练的标签是在8*8的图上进行的loss函数计算
                loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
                loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y + reg_loss_weight*loss_nb_x + reg_loss_weight*loss_nb_y
                #loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y
            else:
                print('No such head:', det_head)
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if det_head == 'pip':
                    print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                    logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                else:
                    print('No such head:', det_head)
                    exit(0)
            epoch_loss += loss.item()
       
        epoch_loss /= len(train_loader)
        if epoch%(save_interval-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            print(filename, 'saved')
        scheduler.step()
    return net

def forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb):
    net.eval()
    with torch.no_grad():
        
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        assert tmp_batch == 1

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)                       #（60,64）
        max_ids = torch.argmax(outputs_cls, 1)                                          #表示矩阵dim=1维度上（每一行）张量最大值的索引
        max_cls = torch.max(outputs_cls, 1)[0]                                
        max_ids = max_ids.view(-1, 1)         
        max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)                              #先复制10倍--->（60,10）再reshape为---> (600,1)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)                           #(60,64)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)                          #取outputs_x横轴最大索引的值    #（60,1）
        outputs_x_select = outputs_x_select.squeeze(1)                                  #（60）
        
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)                           #（60,64）
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)                          #找关键点得分最大索引
        outputs_y_select = outputs_y_select.squeeze(1)                                  #（60）

        outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)              #（600 64）
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)                 #  取的临近关键点都是相对于得分最大格子的索引，因此前面复制了临近关键点数
        outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)           #（60,10）
        outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
        outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)           #（60 10）

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)      #在8*8坐标中的位置加上偏移的量就是关键点的在8*8中的位置（所以此处将坐标中的负数消灭了）
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * input_size[0] / net_stride                                       #（60 1）---除以是为了做归一化
        tmp_y /= 1.0 * input_size[1]/ net_stride

        tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select           #求行的坐标  在8*8坐标中的位置加上偏移的量就是邻近关键点的在8*8中的位置
        tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select          #求列的坐标
        tmp_nb_x = tmp_nb_x.view(-1, num_nb) 
        tmp_nb_y = tmp_nb_y.view(-1, num_nb)
        tmp_nb_x /= 1.0 * input_size[0] / net_stride                                    #需要在归一化的水平上
        tmp_nb_y /= 1.0 * input_size[1] / net_stride                                    #（60,10）

    return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))     
    lms_gt = lms_gt.reshape((-1, 2))         
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm                     #对每个点的距离求均值
    return nme



def MSE_eval_points_crop(lms_pred, lms_gt,norm,image,image_name):
 
    #将坐标回归原图 
    lms_pred_crop=[]
    lms_gt_crop=[]
    for i in range(int(lms_pred.shape[0]/2)):
        x_pred = lms_pred[i*2] * image.shape[1]                                         #小块上的坐标
        x_gt = lms_gt[i*2] * image.shape[1] 
        lms_pred_crop.append(x_pred)
        lms_gt_crop.append(x_gt)
     
        y_pred = lms_pred[i*2+1] * image.shape[0]
        y_gt = lms_gt[i*2+1] * image.shape[0]
        lms_pred_crop.append(y_pred)
        lms_gt_crop.append(y_gt)
        
        cv2.circle(image, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)
        cv2.circle(image, (int(x_gt), int(y_gt)), 1, (255, 0, 0), 1)
        cv2.putText(image, 'Pred', (5,10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, 'GT', (5,30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    outputdir='images/test/resize'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    cv2.imwrite(outputdir +'/'+ image_name, image)
    

    lms_p = np.array(lms_pred_crop).reshape((-1, 2))                  
    lms_g = np.array(lms_gt_crop).reshape((-1, 2))                    
    
    dists = np.mean(np.linalg.norm(lms_p - lms_g, axis=1))/norm
    return dists  

def MSE_eval_points_big(lms_pred, lms_gt,norm,ori,image_name,crop_size):
    
    det_box_scale=1.1
    if True:
        det_xmin = crop_size[0]                          
        det_ymin = crop_size[1]
        det_xmax = crop_size[2]
        det_ymax = crop_size[3]
        
        #找到x,y 中的最大值和最小值，定位出检测目标物框的大小
        det_width = det_xmax - det_xmin
        det_height = det_ymax - det_ymin
        scale = 1.1 
        det_xmin -= int((scale-1)/2*det_width)
        det_ymin -= int((scale-1)/2*det_height)                         #左上角的点进行微调，因为矩形框的宽和高对应放大1.1倍，x向左扩0.1*w/2
        det_width *= scale                                              #宽高放大1.1倍
        det_height *= scale
        det_width = int(det_width)
        det_height = int(det_height)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_width = min(det_width,  ori.shape[1]-det_xmin-1)             #越界判断
        det_height = min(det_height, ori.shape[0]-det_ymin-1)
        
    #pdb.set_trace()
    #将坐标回归原图 
    lms_pred_crop=[]
    lms_gt_crop=[]
   
    for i in range(int(lms_pred.shape[0]/2)):
        x_pred = lms_pred[i*2] * det_width + det_xmin                         
        x_gt = lms_gt[i*2] * det_width+ det_xmin 
        lms_pred_crop.append(x_pred)
        lms_gt_crop.append(x_gt)
     
        y_pred = lms_pred[i*2+1] * det_height + det_ymin
        y_gt = lms_gt[i*2+1] * det_height + det_ymin
        lms_pred_crop.append(y_pred)
        lms_gt_crop.append(y_gt)
        
        cv2.circle(ori, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)
        cv2.circle(ori, (int(x_gt), int(y_gt)), 1, (255, 0, 0), 5)
        cv2.putText(ori, 'Pred', (5,30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(ori, 'GT', (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    outputdir='images/test/ori'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    cv2.imwrite(outputdir +'/'+ image_name, ori)
    
    lms_p = np.array(lms_pred_crop).reshape((-1, 2))                  
    lms_g = np.array(lms_gt_crop).reshape((-1, 2))                    
    
    dists = np.mean(np.linalg.norm(lms_p - lms_g, axis=1))/norm   
    print ('dists is :',dists)
    
    return dists 



def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc
