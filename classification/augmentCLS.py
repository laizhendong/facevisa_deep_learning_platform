# -*- coding: utf-8 -*-
import cv2
import json
import random
from tqdm import tqdm
import utils.function as F
import os
import argparse
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
                aug = F.Flip(*values).apply
                aug_list.append(aug)
            if key == "Affine":
                aug = F.Affine(*values).apply
                aug_list.append(aug)
            if key == "Resize":
                aug = F.Resize(*values).apply
                aug_list.append(aug)
    return aug_list


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-cfg",help="location of arguments.json",default="/arguments/arguments.json")
    ap.add_argument("-txt",help="location of dataset txt",default="/data/train_shuffle.txt")
    args = ap.parse_args()
    with open(args.cfg, "r", encoding="utf-8") as f:
        content = json.load(f)
    if len(content["share_transforms"]) > 0 or len(content["offset_transforms"]) > 0:
        en_list = []
        f = open(args.txt)
        files = f.readlines()
        for file in tqdm(files):
            path_cls = file.strip('\n').split(' ')
            #file_path = path_cls[0][:12]
            #name = path_cls[0][11:]
            #cls = path_cls[1]
            data_path = path_cls[0]
            print(data_path)
            if os.path.exists(data_path)==False:
                print("no find img in this path")
                continue
            img = cv2.imread(data_path)
            if len(content["share_transforms"]) > 0:
                share_aug_list = share_transforms(content["share_transforms"])
                img = F.pipeline(img, share_aug_list)
            if len(content["offset_transforms"]) > 0:
                offset_aug_list = offset_transforms(content["offset_transforms"])
                img = F.pipeline(img, offset_aug_list)
            cv2.imwrite(data_path[:-4] + '_en.jpg', img)
            print("save img_en !!")
            path_cls1 = file.split(' ')
            en_list.append(path_cls1[0][:-4] + '_en.jpg'+" "+path_cls1[1])
        f.close()
        fw = open(args.txt, 'a')
        fw.writelines(en_list)
        fw.close()
    else:
        print("no data_augment!")





 
