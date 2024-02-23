# -*- coding: utf-8 -*-
import cv2
import json
import random
from tqdm import tqdm
import function as F
import os


def share_transforms(params: list = []):
    aug_list = []
    for i in params:
        key = i["type"]
        values = list(i["params"].values())
        p = i["probability"]
        if p > random.random():
            if key == "Blur":
                aug = F.Blur(*values).apply
                aug_list.append(aug)
            if key == "MedianBlur":
                aug = F.MedianBlur(*values).apply
                aug_list.append(aug)
            if key == "MotionBlur":
                aug = F.MotionBlur(*values).apply
                aug_list.append(aug)
            if key == "GaussianBlur":
                aug = F.GaussianBlur(*values).apply
                aug_list.append(aug)
            if key == "Sharpen":
                aug = F.Sharpen(*values).apply
                aug_list.append(aug)
            if key == "GaussNoise":
                aug = F.GaussNoise(*values).apply
                aug_list.append(aug)
            if key == "ISONoise":
                aug = F.ISONoise(*values).apply
                aug_list.append(aug)
            if key == "MultiplicativeNoise":
                aug = F.MultiplicativeNoise(*values).apply
                aug_list.append(aug)
            if key == "RGBShift":
                aug = F.RGBShift(*values).apply
                aug_list.append(aug)
            if key == "RandomBrightnessContrast":
                aug = F.RandomBrightnessContrast(*values).apply
                aug_list.append(aug)
            if key == "ChannelShuffle":
                aug = F.ChannelShuffle().apply
                aug_list.append(aug)
            if key == "ToGray":
                aug = F.ToGray().apply
                aug_list.append(aug)
            if key == "CLAHE":
                aug = F.CLAHE(*values).apply
                aug_list.append(aug)
            if key == "RandomGamma":
                aug = F.RandomGamma(*values).apply
                aug_list.append(aug)
            if key == "Dilation":
                aug = F.Dilation(*values).apply
                aug_list.append(aug)
            if key == "Erosion":
                aug = F.Erosion(*values).apply
                aug_list.append(aug)
    return aug_list


def offset_transforms(params: list = []):
    aug_list = []
    for i in params:
        key = i["type"]
        values = list(i["params"].values())
        p = i["probability"]
        if p > random.random():
            if key == "Flip":
                aug = F.Flip(*values).apply_to_bbox
                aug_list.append(aug)
            if key == "Affine":
                aug = F.Affine(*values).apply_to_bbox
                aug_list.append(aug)
            if key == "Resize":
                aug = F.Resize(*values).apply_to_bbox
                aug_list.append(aug)
    return aug_list


if __name__ == "__main__":
    with open("/arguments/arguments.json", "r", encoding="utf-8") as f:
        content = json.load(f)
    print(content)
    img_path = "/datasets/yolo_data/stain/images/train/"
    img_outpath = "/datasets/yolo_data/stain/images/train/"
    label_path = "/datasets/yolo_data/stain/labels/train/"
    label_outpath = "/datasets/yolo_data/stain/labels/train/"
    img_list = os.listdir(img_path)
    if len(content["share_transforms"]) > 0 or len(content["offset_transforms"]) > 0:
        for img_name in tqdm(img_list):
            print(img_name)
            if os.path.exists(img_path + img_name)==False:
                print("no find img in this path")
                continue
            img = cv2.imread(img_path + img_name)
            label = open(label_path + img_name[:-4] + '.txt').readlines()
            targets = []
            bboxes_out = []
            for bbox in label:
                bbox = bbox.split(" ")
                _, x1, y1, x2, y2 = bbox[:5]
                targets.append([int(_), float(x1), float(y1), float(x2), float(y2)])
            if len(content["share_transforms"]) > 0:
                share_aug_list = share_transforms(content["share_transforms"])
                img = F.pipeline(img, share_aug_list)
            if len(content["offset_transforms"]) > 0:
                offset_aug_list = offset_transforms(content["offset_transforms"])
                (img, targets) = F.pipeline((img, targets), offset_aug_list)
            cv2.imwrite(img_outpath + img_name[:-4] + '_en.jpg', img)
            with open(label_outpath + img_name[:-4] + '_en.txt', 'w') as f:
                for line in targets:
                    f.write(" ".join([str(a) for a in line]) + "\n")
                f.close()
    else:
        print("no data_augment!")






 
