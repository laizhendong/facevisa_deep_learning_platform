# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os
from os import getcwd
import shutil
import cv2
import argparse
import yaml


curPath = os.path.dirname(os.path.realpath(__file__))
# 获取yaml文件路径
yamlPath = os.path.join(curPath, "test_data.yaml")
 
# open方法打开直接读出来
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
# 用load方法转字典
label_dict = yaml.load(cfg)  
#print(label_dict['name'])
#print(label_dict['nc'])

#sets = ['train','val','test']
#classes = ['apple', 'orange', 'banana']  #类别数目
sets = label_dict['dataset']
classes = label_dict['names']
abs_path = os.getcwd()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(voc_imgxml_path + r'/%s.xml' % (image_id))
    out_file = open(yolopath + '/labels/'+ image_set+'/'+ '%s.txt' % (image_id.split('/')[-1]), 'a')
    #print(image_id)
    if(os.path.isfile(voc_imgxml_path + '/%s.bmp' % (image_id))):
        img = cv2.imread(voc_imgxml_path+ '/%s.bmp' % (image_id))
    elif(os.path.isfile(voc_imgxml_path + '/%s.png' % (image_id))): 
        img = cv2.imread(voc_imgxml_path+ '/%s.png' % (image_id))
    else:
        img = cv2.imread(voc_imgxml_path + '/%s.jpg' % (image_id))

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = img.shape[1]
    h = img.shape[0]
    #w = int(size.find('width').text)
    #h = int(size.find('height').text)
    print(w,h)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        print(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



parser = argparse.ArgumentParser()
parser.add_argument('--yolopath', dest='yolo_path',help='yolo dataset',default='/test_datasets/yolo_data/stain', type=str)
parser.add_argument('--vocpath', dest='voc_path',help='voc dataset',default='/test_datasets/voc_data/stain', type=str)
parser.add_argument('--classes', dest='classes',help='classes to train exclude bg',default=label_dict['names'], nargs='+',type=str)

args = parser.parse_args()
yolopath = args.yolo_path
vocpath = args.voc_path
classes = args.classes
print(args)

wd = getcwd()
for image_set in sets:
    if not os.path.exists(yolopath + '/labels/'+ image_set):
        os.makedirs(yolopath + '/labels/'+ image_set)
    if not os.path.exists(yolopath + '/images/'+ image_set):
        os.makedirs(yolopath + '/images/'+ image_set)
    voc_imgxml_path = vocpath+ '/imgxml/'
    voc_ImageSets_path = vocpath+ '/ImageSets/'
    image_ids = open(voc_ImageSets_path+ '/%s.txt' % (image_set)).read().strip().split()
    list_file = open(yolopath + '/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        print(image_ids.index(image_id),"   ", image_id)
        if(os.path.isfile(voc_imgxml_path + '/%s.bmp' % (image_id))):
            list_file.write(voc_imgxml_path + '/%s.bmp\n' % (image_id))
            shutil.copy(voc_imgxml_path + '/%s.bmp' % (image_id), yolopath + '/images/'+ image_set+'/'+ '%s.bmp' % (image_id.split('/')[-1]))
            
        elif(os.path.isfile(voc_imgxml_path + '/%s.png' % (image_id))):  
            list_file.write(voc_imgxml_path + '/%s.png\n' % (image_id))
            shutil.copy(voc_imgxml_path + '/%s.png' % (image_id), yolopath + '/images/'+ image_set+'/'+ '%s.png' % (image_id.split('/')[-1]))
            
        else:
            list_file.write(voc_imgxml_path + '/%s.jpg\n' % (image_id))
            shutil.copy(voc_imgxml_path + '/%s.jpg' % (image_id), yolopath + '/images/'+ image_set+'/'+ '%s.jpg' % (image_id.split('/')[-1]))
        convert_annotation(image_id)
    list_file.close()

