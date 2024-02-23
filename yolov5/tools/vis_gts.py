# -*- coding: utf-8 -*-
import os
import os.path as osp
import random
import shutil
import cv2
import numpy as np
"""
Author:pengfei
根据：yolo格式的标注，保存bbox的可视化结果，保存指定目录下
"""

def vis_gts():
    label_dir = "/home_disk2/caffe/dty/sideBottomCenter_tailExtra/data/labels/train"
    im_dir = "/home_disk2/caffe/dty/sideBottomCenter_tailExtra/data/images/train"
    num_to_save = 1000
    
    
    save_dir = './vis_save/'
    
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
                elif box[i] > 2047:
                    box[i] = 2047

            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))
            cv2.putText(im, CLS_LIST[int(obj_cls)], (box[0], box[1]), 2,2, (255, 0, 0))

        cv2.imwrite(osp.join(save_dir, im_name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])



if __name__ == '__main__':
    vis_gts()