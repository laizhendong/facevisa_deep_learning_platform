# -*- coding: utf-8 -*-
import os
import os.path as osp
import random
import shutil
import cv2
import numpy as np
import argparse
"""
Author:pengfei
根据：yolo格式的标注，保存bbox的可视化结果，保存指定目录下
"""

def vis_gts(label_dir,im_dir,save_dir):
    #label_dir = "/data1/lilai/XXjiasi51_56/YOLO/labels/trainval"
    #im_dir = "/data1/lilai/XXjiasi51_56/YOLO/images/trainval"
    num_to_save = 100
    
   
    #save_dir = './vis_save/'
    
    print(" read label from {} \n read image from {}\n save mark into {} ".format(label_dir, im_dir, num_to_save))
    
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    name_list = os.listdir(label_dir)
    random.shuffle(name_list)

    CLS_LIST = ['js', 'ws']

    for label_name in name_list[:num_to_save]:

        im_name = label_name[:-3] + 'jpg'
        # print(osp.join(label_dir, label_name))
        with open(osp.join(label_dir, label_name)) as f:
            all_gts = f.readlines()

        im = cv2.imread(osp.join(im_dir, im_name))
        im_h, im_w = im.shape[:2]

        for i_gt in all_gts:
            obj_cls = i_gt.strip().split(" ")[0]

            # covert yolo-format coord to (xmin ymin xmax ymax)
            box = list(map(float, i_gt.strip().split(" ")[1:]))
            box = np.array(box)
            box[::2] = box[::2] * im_w
            box[1::2] = box[1::2] * im_h
            box[0], box[1], box[2], box[3] = box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2
            box = box.astype(np.int16)
            for i in range(4):
                if box[i] < 0:
                    box[i] = 0
                elif box[i] > im_w:
                    box[i] = im_w

            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            cv2.putText(im, CLS_LIST[int(obj_cls)], (box[0], box[1]), 2,2, (255, 0, 0))

        cv2.imwrite(osp.join(save_dir, im_name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', dest='yolo_label',help='yolo label to train on',default='trainval', type=str)
    parser.add_argument('--img', dest='yolo_img',help='yolo img to train on',default='trainval', type=str)
    parser.add_argument('--savedir', dest='save_dir',help='marked label on img',default='./vis_save', type=str)
    args = parser.parse_args()
    
    vis_gts(args.yolo_label, args.yolo_img, args.save_dir)