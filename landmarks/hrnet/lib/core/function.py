# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
from  torchvision import utils as vutils
from core.evaluate import accuracy,MSE_eval_points_small,MSE_eval_points_big
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images,save_batch_joints_image_pts,save_batch_final_joints_image_pts
import cv2
import pdb 


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)
        
        # compute output
        outputs = model(input)
       
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
                
        else:
            output = outputs
            loss = criterion(output, target, target_weight)
           

        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            
            #'/t,表示4个空格，这个速度不知道在算个啥：样本数/时间?'

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            
            if config.DEBUG.SAVE_IMAGE_PTS:    #保存pts结果图
                    save_batch_joints_image_pts(meta, input, pred*4.0, output_dir+'/trainpts/outpts/')
                    #save_batch_final_joints_image_pts(meta, preds, output_dir+'/trainpts/finaloutpts/')
                    
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    dist_big= AverageMeter()
    dist_small= AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    num_samples = len(val_dataset)
    #B*16*3
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                
                #左右对称的关键点进行坐标交换
                output_flipped = flip_back(output_flipped.cpu().numpy(),       
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                #取0-63列的数据，替换到1-63列的数据
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]   
                
                #输出图片取均值
                output = (output + output_flipped) * 0.5           

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            #criterion_2=torch.nn.SmoothL1Loss().cuda()
            loss = criterion(output, target, target_weight)
            #loss = criterion_2(target, output)
            #loss = criterion(output, target)

            num_images = input.size(0)   # batchsize
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

           
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            
            
            #根据输出的图得到预测的结果：含最大值的索引和最大索引所含的值
            #输出仿射逆变换之后的图 ，输入热图
            #pdb.set_trace()
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)           

            
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)         #横轴上连乘
            
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            
            #print('pic_name is :',meta['image'])
            
            distbig= MSE_eval_points_big( preds, meta)                     #回归原图之后坐标点做差
            #print('dist_big is ',distbig)
            distsmall = MSE_eval_points_small( pred*4, meta)               #输入模型大小
            #print('dist_small is ',distsmall)
            
            dist_big.update(distbig)
            dist_small.update(distsmall)
            
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                
                #保存pts结果图
                if config.DEBUG.SAVE_IMAGE_PTS:    
                    save_batch_joints_image_pts(meta, input, pred*4.0, output_dir+'/outpts/')
                    save_batch_final_joints_image_pts(meta, preds, output_dir+'/out/')
                
                #此处的pred*4将坐标点还原到输入网络的大小
                save_debug_images(config, input, meta, target, pred*4.0, output,
                                  prefix)         

                
        #name_values, perf_indicator = val_dataset.evaluate(
            #config, all_preds, output_dir, all_boxes, image_path,
            #filenames, imgnums
        #)

        #model_name = config.MODEL.NAME
        #if isinstance(name_values, list):
         #   for name_value in name_values:
                #_print_name_value(name_value, model_name)
        #else:
          #  _print_name_value(name_values, model_name)

        #tensorboard可视化
        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            #if isinstance(name_values, list):
            #    for name_value in name_values:
             #       writer.add_scalars(
              #          'valid',
               #         dict(name_value),
                #        global_steps
                 #   )
            #else:
             #   writer.add_scalars(
              #      'valid',
               #     dict(name_values),
               #     global_steps
               # )
            writer_dict['valid_global_steps'] = global_steps + 1

    #return perf_indicator
    logger.info('dist_big:{dist_big.avg:.3f}\t'.format(dist_big=dist_big))
    logger.info('dist_small:{dist_small.avg:.3f}\t'.format(dist_small=dist_small))
  
    return 0


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
