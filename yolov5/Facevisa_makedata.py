import os
import sys
import random
import numpy as np
import pandas as pd 
from tqdm.auto import tqdm
import shutil as sh
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element

file_ext = ['jpg', 'png', 'bmp']
# data-paper
obj_names = ['break', 'stain', 'hollow', 'fold', 'tube']
num_split = 6

def file_search(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            file_search(filepath, allfile)
        else:
            if filepath.split(".")[-1] in file_ext:
                allfile.append(filepath.strip())
    return allfile

def xml_parse(xml_name):
    tree = ET.parse(xml_name)
    #root = tree.getroot()
    im_info = tree.find('size')
    im_w = float(im_info.find('width').text)
    im_h = float(im_info.find('height').text)
    im_d = im_info.find('depth').text
    
    nodes = tree.findall('object')
    bndboxes = []
    bndnames = []
    for child in nodes:
    
        name = child.find('name').text
        if name not in obj_names:
            continue
        bndnames.append(name)
        
        bndbox = []
        box = child.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        x_center = xmin + 0.5*w
        y_center = ymin + 0.5*h
        bndbox.append(float(x_center))
        bndbox.append(float(y_center))
        bndbox.append(float(w))
        bndbox.append(float(h))
        bndbox = np.array(bndbox, np.int32)
        bndboxes.append(bndbox)
        
    return im_w, im_h, im_d, bndboxes, bndnames


# index = []
# files_folder = r'/home2/baseline_det/VOCdevkit2007/VOC2007/JPEGImages'
# file_search(files_folder, index)
# random.shuffle(index)

index = []
txt_trainval = "/home2/baseline_det/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt"
txt_train = "/home2/baseline_det/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt"
txt_val = "/home2/baseline_det/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt"

f_train = open(txt_train, "w")
f_val = open(txt_val, "w")

with open(txt_trainval, "r") as f:
    lines = f.readlines()
for line in lines:
    img_info = "/home2/baseline_det/VOCdevkit2007/VOC2007/JPEGImages/" + line.strip() + ".jpg"
    index.append(img_info)
random.shuffle(index)

print(" num samples: ", len(index))

source = 'train'
if True:
    # for fold in [0,1,2,3,4]:
    for fold in [0]:
        val_index = index[len(index)*fold//num_split:len(index)*(fold+1)//num_split]
        for name in tqdm(index):
            if name in val_index:
                path2save = 'val2017/'
                f_val.write(name + "\n")
            else:
                path2save = 'train2017/'
                f_train.write(name + "\n")
            if not os.path.exists('convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('convertor/fold{}/labels/'.format(fold)+path2save)
            
            xml_file = name.split("/")
            xml_file[-1] = xml_file[-1].replace(".jpg",".xml")
            xml_name = '/'.join(xml_file).replace("JPEGImages", "Annotations")
            im_w, im_h, im_d, boxes, names = xml_parse(xml_name)
            if len(boxes) != len(names):
                print(f"Error: {xml_name}")
                continue
            with open('convertor/fold{}/labels/'.format(str(fold))+path2save+ name.split("/")[-1].replace(".jpg",".txt"), 'w+') as f:
                for idx, obj_name in enumerate(names):
                    classes = obj_names.index(obj_name)
                    x_center = boxes[idx][0]/im_w
                    y_center = boxes[idx][1]/im_h
                    w = boxes[idx][2]/im_w
                    h = boxes[idx][3]/im_h
                    row = [classes,x_center,y_center,w,h]
                    text = ' '.join([str(i) for i in row])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy(name,'convertor/fold{}/images/{}/{}'.format(fold,path2save,name.split("/")[-1]))

f_train.close()
f_val.close()