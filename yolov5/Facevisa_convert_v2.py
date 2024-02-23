#!/usr/bin/env python
# -*- coding:utf-8 -*-


__author__ = "Facevisa_lilai"

import os
import sys
import struct
import torch
import pandas as pd

from torch import nn
from torch.nn import functional as F
import torchvision
from torchsummaryX import summary
import yaml

# 加载平台设置的参数配置文件：alg_params.yaml
curPath = os.path.dirname(os.path.realpath(__file__))
# 获取yaml文件路径
yamlPath = os.path.join(curPath, "alg_params.yaml")
# open方法打开直接读出来
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
# 用load方法转字典
train_dict = yaml.load(cfg)  

def summary_model(model:torch.nn.Module, related_layer_types=(nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
              
    def parse_param_bn(module):
        layer_param = []
        layer_param.append(module.num_features )
        layer_param.append(module.eps)
        return layer_param
                 
    def parse_param_fc(module):
        layer_param = []
        layer_param.append(module.in_features )
        layer_param.append(module.out_features)

        if type(module.bias) == nn.parameter.Parameter: 
            layer_param.append(1)
        else:
            layer_param.append(0)
        return layer_param
            
    def parse_param_conv(module):
        layer_param = []
        layer_param.append(module.in_channels)
        layer_param.append(module.out_channels)
        
        if type(module.kernel_size) == int:
            layer_param.append(module.kernel_size)
            layer_param.append(module.kernel_size)
        elif type(module.kernel_size) == tuple:
            layer_param.append(module.kernel_size[0])
            layer_param.append(module.kernel_size[1])
            
        if type(module.stride) == int: 
            layer_param.append(module.stride)
            layer_param.append(module.stride)
        elif type(module.stride) == tuple:
            layer_param.append(module.stride[0])
            layer_param.append(module.stride[1])

        if type(module.padding) == int: 
            layer_param.append(module.padding)
            layer_param.append(module.padding)
        elif type(module.padding) == tuple:
            layer_param.append(module.padding[0])
            layer_param.append(module.padding[1])            
 
        if type(module.dilation) == int: 
            layer_param.append(module.dilation)
            layer_param.append(module.dilation)
        elif type(module.dilation) == tuple:
            layer_param.append(module.dilation[0])
            layer_param.append(module.dilation[1])     

        layer_param.append(module.groups)
        if module.bias is None:
            layer_param.append(0)
        else:
            layer_param.append(1)
        
        return layer_param

    
    info = []
    info_param = {}
    for name, module in model.named_modules():
        if type(module) == nn.Conv2d:
            info.append({"name": name, "module": module})
            info_param[name+".weight"] = parse_param_conv(module)
        elif type(module) == nn.BatchNorm2d:
            info.append({"name": name, "module": module})
            info_param[name+".weight"] = parse_param_bn(module)             
        elif type(module) == nn.Linear:
            info.append({"name": name, "module": module})
            info_param[name+".weight"] = parse_param_fc(module)  


    df = pd.DataFrame(info)
    df = df.reindex(columns=["name", "module"])
    #print(df.to_markdown())
    
    return info_param

def Facevisa_info_count(net_params):
    total_info_num = 0
    if net_params.get("hwc2chw"):
        total_info_num += 1
    if net_params.get("bgr2rgb"):  
        total_info_num += 1    
    if net_params.get("vid255"):  
        total_info_num += 1 
    if net_params.get("mean"):  
        total_info_num += 1    
    if net_params.get("std"):  
        total_info_num += 1    
    return total_info_num

def Facevisa_weight_Convert(net_params, weight, wts_name):

    # check operation
    total_info_num = Facevisa_info_count(net_params)
    
    # load
    device = torch.device('cpu')
    model = torch.load(weight, map_location=device)['model'].float()
    model.to(device).eval()
    
    # summary 
    net_info = summary_model(model)
 
    # save
    f = open(wts_name, 'w')
    f.write("{}\n".format(len(model.state_dict().keys()) + len(net_info) + total_info_num))
    
    # fill preprocess info 
    if net_params.get("hwc2chw"):
        f.write("{} {}".format("hwc2chw", 1))
        if net_params["hwc2chw"]:
            f.write(" ")
            f.write(struct.pack(">f", float(1.0)).hex())
        else:
            f.write(" ")
            f.write(struct.pack(">f", float(0.0)).hex())
        f.write("\n")
        
    if net_params.get("bgr2rgb"):  
        f.write("{} {}".format("bgr2rgb", 1))
        if net_params["bgr2rgb"]:
            f.write(" ")
            f.write(struct.pack(">f", float(1.0)).hex())
        else:
            f.write(" ")
            f.write(struct.pack(">f", float(0.0)).hex())
        f.write("\n")
        
    if net_params.get("vid255"):  
        f.write("{} {}".format("vid255", len(net_params["vid255"])))
        for vv in net_params["vid255"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    if net_params.get("mean"):  
        f.write("{} {}".format("mean", len(net_params["mean"])))
        for vv in net_params["mean"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

    if net_params.get("std"):      
        f.write("{} {}".format("std", len(net_params["std"])))
        for vv in net_params["std"]:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
    
    for k,v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        # record weight
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")
        # record info
        if net_info.get(k):
            f.write("{}_info {}".format(k, len(net_info[k])))
            for idx in net_info[k]:
                f.write(" ")
                f.write(struct.pack(">f", float(idx)).hex())
            f.write("\n")

if __name__ == '__main__':
    # If there is no corresponding operation, delete the corresponding key
    net_params = {}
    net_params["N"] = 1
    net_params["C"] = 3
    net_params["H"] = max(train_dict['img_size'])#608
    net_params["W"] = max(train_dict['img_size'])#608
    net_params["CLS"] = 2#80
    net_params["hwc2chw"] = True
    net_params["bgr2rgb"] = True                 # 需要进行BGR2RGB是因为训练的时候 最终输入的格式是RGB 。 见utils/datasets.py 593代码 img = img[:, :, ::-1].transpose(2, 0, 1)
    net_params["vid255"] = [255.0, 255.0, 255.0] # 除以255的原因是因为 训练的预处理强制了这一步 见train.py 244行
    net_params["mean"] = [0.0, 0.0, 0.0]
    net_params["std"] = [1.0, 1.0, 1.0]   
    net_params["valid"] = False
    net_params["show_info"] = False

    # weight file
    # weight = r'weights/yolov5s.pt'
    # wts_name = r'weights/yolov5s.wts'
    weight = r'/output/exp0/weights/last.pt'
    wts_name = r'/output/exp0/weights/last.wts'

    # convert weight
    print("\n\n----------------weight convert--------------------\n\n")        
    Facevisa_weight_Convert( net_params, weight, wts_name)


