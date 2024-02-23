# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np
import json
import os
from PIL import Image
import os.path
import shutil
import string

def readbbox(xml):
    tree = ET.parse(xml)
    # root = tree.getroot()
    nodes = tree.findall('object')
    bndboxes = []
    for child in nodes:
        bndbox = []
        tmp = child.find('bndbox')
        xmin = tmp.find('xmin').text
        ymin = tmp.find('ymin').text
        xmax = tmp.find('xmax').text
        ymax = tmp.find('ymax').text
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)
        bndbox = np.array(bndbox, np.int32)
        bndboxes.append(bndbox)
    return bndboxes

def if_has_shape(xml):
    tree = ET.parse(xml)
    # root = tree.getroot()
    nodes = tree.findall('object')
    shapes = []
    tmp = []
    for child in nodes:
        shape = []
        tmp = child.find('shape')
    return tmp

def file_extension(path):
    return os.path.splitext(path)[1]

def readshape(xml):
    tree = ET.parse(xml)
    # root = tree.getroot()
    nodes = tree.findall('object')
    shapes = []
    for child in nodes:
        shape = []
        tmp = child.find('shape')
        tmp_shape = tmp.find('points')
        for n in range(0, len(tmp_shape)):
            value = tmp_shape[n].text
            shape.append(value)
        shape = np.array(shape, np.int32)
        shapes.append(shape)
    return shapes

def iou(box, clusters):
    """
   计算 IOU
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    print(box)
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        print(x,y)
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


#  计算框的 numpy 数组和 k 个簇之间的平均并集交集（IoU）。
def avg_iou(boxes, clusters):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


# 将所有框转换为原点。
def translate_boxes(boxes):
    """
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


# 使用联合上的交集（IoU）度量计算k均值聚类。
def kmeans(boxes, k, dist=np.median):
    """
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]  # 初始化k个聚类中心（方法是从原始数据集中随机选k个）

    while True:
        for row in range(rows):
            # 定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid)。到聚类中心的距离越小越好，但IOU值是越大越好，所以使用 1 - IOU，这样就保证距离越小，IOU值越大。
            distances[row] = 1 - iou(boxes[row], clusters)
        # 将标注框分配给“距离”最近的聚类中心（也就是这里代码就是选出（对于每一个box）距离最小的那个聚类中心）。
        nearest_clusters = np.argmin(distances, axis=1)
        # 直到聚类中心改变量为0（也就是聚类中心不变了）。
        if (last_clusters == nearest_clusters).all():
            break
        # 更新聚类中心（这里把每一个类的中位数作为新的聚类中心）
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


# 获取图片宽高
def get_image_width_high(full_image_name):
    image = Image.open(full_image_name)
    image_width, image_high = image.size[0], image.size[1]
    return image_width, image_high


def get_kmeans(bboxes, cluster_num=9):
    anchors = kmeans(bboxes, cluster_num)
    ave_iou = avg_iou(bboxes, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou

if __name__ == "__main__":
    xml_srcdir = r"/data1/LZD/YOLOV5/YOLO_Stiff/"
    bboxes = []
    for rootdir, dirs, files in os.walk(xml_srcdir):
        for name in files:
            print(rootdir,name)
            filepath = os.path.join(rootdir, name)
            if file_extension(filepath) == '.xml':
                tree = ET.parse(filepath)
                root = tree.getroot()
                # root.find('filename').text = xmlFile.replace('xml','jpg')
                print(filepath)
                for Object in root.findall('object'):
                    bbox = Object.find('bndbox')
                    #x_left = int(bbox.find('xmin').text)
                    #y_top = int(bbox.find('ymin').text)
                    width = int(bbox.find('xmax').text) - int(bbox.find('xmin').text)
                    height = int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
                    bboxes.append([width,height])
    anchors, ave_iou = get_kmeans(np.array(bboxes), cluster_num=9)
    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]
    print('anchors are: {anchor_string}', anchor_string)
    with open("anchor.txt", 'a') as f:
        f.write(anchor_string +"     ")
        f.write(str(ave_iou) + "\n")
    
    print('the average iou is: {ave_iou}', ave_iou)

