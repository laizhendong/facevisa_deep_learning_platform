# -*- coding: utf-8 -*-
#from typing import OrderedDict


from copy import deepcopy
import torch
from symbol import resnet,wideresnet
from utils import optional_file
from utils.modelcard import MODELCARD
try:
    import onnxruntime
    import onnx
    flag_check_onnx = True
except ImportError as e:
    flag_check_onnx = False
import os.path as osp
import numpy as np
from symbol.repvgg import get_RepVGG_func_by_name, func_dict
from config.default_config import get_default_config
import argparse
import os
from collections import OrderedDict
from utils import experiments

from utils.torch_utils import select_device


def load_model_with_eval(backbone_name,num_classes, model_name, input_shape):

    if backbone_name.startswith('resnet'):
        torch_net = resnet.RESNET(backbone_name = backbone_name, input_channels = input_shape[1], num_classes=[num_classes], pretrained_backbone=False, with_softmax=True)
        torch_net.load_state_dict(torch.load(model_name), strict=False)
    elif backbone_name.startswith("wideresnet"):
        torch_net = wideresnet.WRESNET(backbone_name,num_classes=num_classes,with_softmax=True)
        checkpoint = torch.load(model_name)
        if "ema_model" in checkpoint.keys() and "model" in checkpoint.keys():
            ema_load = checkpoint['ema_model'] #model from torchssl
            #model_load = checkpoint['model']
            model_load = None
            print("[I] find ema_model")
        else:
            ema_load = None
            model_load = checkpoint['model'] #model from torchssl
            
        # model_load_remap = OrderedDict()
        # for name,v in model_load.items():
        #     if name.startswith("module."): 
        #         newname = '.'.join(name.split('.')[1:])
        #         model_load_remap[newname] = v 
        #     else:
        #         model_load_remap = v
        # torch_net.load_state_dict(model_load_remap, strict=True) 
            
        if not (ema_load is None):
            print("[I] load ema_model")
            ema_load_remap = OrderedDict()
            for name,v in ema_load.items():
                if name.startswith("module."): 
                    newname = '.'.join(name.split('.')[1:])
                    ema_load_remap[newname] = v 
                else:
                    ema_load_remap = v
            torch_net.load_state_dict(ema_load_remap, strict=True) 
        elif not (model_load is None):
            print("[I] load model")
            model_load_remap = OrderedDict()
            for name,v in model_load.items():
                if name.startswith("module."): 
                    newname = '.'.join(name.split('.')[1:])
                    model_load_remap[newname] = v 
                else:
                    model_load_remap = v
            torch_net.load_state_dict(model_load_remap, strict=True)    
                        
    elif backbone_name.lower().startswith('repvgg'):
        repvgg_model_list = list(func_dict.keys())
        if not backbone_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, backbone_name))

        repvgg_build_func = get_RepVGG_func_by_name(backbone_name)
        torch_net = repvgg_build_func(deploy=True, num_classes=num_classes, with_softmax=True) # dataset_test.class_num()
        torch_net.load_state_dict(torch.load(model_name))

    
    torch_net = torch_net.cuda().eval()

    return torch_net

def export_to_onnx(net, input_shape, save_name, tensors_name, num_classes, is_dynamic_shape=False):
    if not osp.exists(model_name):
        raise Exception("can not find model {}".format(model_name))

    if (not type(tensors_name['input']) is list)or (not type(tensors_name['output']) is list):
        raise Exception("type of input/output must be  list, please check")


    n, c, h, w = input_shape
    dummy_input = torch.randn(n, c, h, w, device='cuda')

    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    input_names = tensors_name["input" ] 
    output_names = tensors_name["output"]
    
    if is_dynamic_shape:
        dynamic_shape = {input_names[0]:{0 : 'batch_size'}, output_names[0]: {0 : 'batch_size'}}
        print("  export to onnx with dynamic shape")
        torch.onnx.export(net, dummy_input, save_name, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_shape)
    else:
        print("  export to onnx with fixed shape {}".format(input_shape))
        torch.onnx.export(net, dummy_input, save_name, verbose=False, input_names=input_names, output_names=output_names)


def check_onnx(onnx_name):
    """
    verify the model’s structure and confirm that the model has a valid schema. 
    The validity of the ONNX graph is verified by checking the model’s version, 
    the graph’s structure, as well as the nodes and their inputs and outputs.
    """
    onnx_model = onnx.load(onnx_name)
    onnx.checker.check_model(onnx_model)


def compare_onnx_pth(torch_net, input_shape, onnx_name):
    """
    """
    # random input tensor 
    n, c, h, w = input_shape
    input_tensor = torch.randn(n, c, h, w, device='cuda')

    torch_outs = torch_net(input_tensor)

    ort_session = onnxruntime.InferenceSession(onnx_name)

    def to_numpy(tensor):
        if isinstance(tensor,list):
            return [t.detach().cpu().numpy() if t.requires_grad else t.cpu().numpy() for t in tensor]
        else:
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
    onnx_outs = ort_session.run(None, onnx_inputs)[0]

    try:
        if isinstance(torch_outs,list):
            for torch_out,onnx_out in zip(torch_outs, onnx_outs):
                np.testing.assert_allclose(to_numpy(torch_out), onnx_out, rtol=1e-03, atol=1e-03)
        else:
            np.testing.assert_allclose(to_numpy(torch_outs), onnx_outs, rtol=1e-03, atol=1e-03)
        print("   Exported model has been tested with ONNXRuntime, and the result looks good!", 'blue')
    except:
        print("   The two outputs are inconsistent!", 'red')
        raise Exception()




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_config",help="exp configuration file")
    ap.add_argument("-card_file",help="model card",required=True)
    args = ap.parse_args()
    exp_cfg = experiments.load_config(args.exp_config,'to_onnx')

    cfg = get_default_config()
    cfg.merge_from_file(exp_cfg['hyparam_file'])
    cfg = optional_file.load_optional_file(cfg,exp_cfg['optional_file'])
    cfg.freeze()
    print("------------configuration after merge--------------")
    print(cfg)
    modelcard = MODELCARD(task='classification',version=1)
    with open(args.card_file,'r',encoding='utf-8') as f:
        json_data = f.read()
        modelcard.fromjson(json_data)
    print(modelcard.tojson())
    
    # set params for convert
    backbone_name = cfg.MODEL.BACKBONE_NAME # resnet50
    B,C,H,W = modelcard.input_shape
    use_dynamic_shape =  True if B < 0 else False
    B = 1 if B < 0 else B
    input_shape = [B,C,H,W]
     
    num_classes =  len(modelcard.classes_local2global.keys())

    work_dir = os.getcwd()
    model_name = os.path.splitext(args.card_file)[0]
    saved_name = os.path.splitext(model_name)[0] + ".onnx"       


 
    # 输入输出tensor, 这里面输入tensor命名为data， 输出tensor命名为prob
    tensors_name = {'input': modelcard.input_blobs,'output':modelcard.output_blobs}
    # --------------------------------------------------------------
    print("backbone: {} \ninput shape: {} \nnum_classes: {} \nmodel_name: {}".format(backbone_name, input_shape, num_classes, model_name))
    print("saved_name: {} \nuse_dynamic_shape: {} \ntensors_name: {}".format(saved_name, use_dynamic_shape, tensors_name))
    
    # load torch model
    torch_net = load_model_with_eval(backbone_name, num_classes, model_name, input_shape)
   
    device = select_device("0") 
    torch_net.to(device=device)

    # convert to onnx 
    export_to_onnx(net = torch_net, input_shape = input_shape,  save_name = saved_name, 
                    tensors_name = tensors_name, num_classes=num_classes, is_dynamic_shape=use_dynamic_shape)
    with open(saved_name + ".json",'w',encoding='utf-8') as f:
        f.write(modelcard.tojson())

    if flag_check_onnx:
        check_onnx(onnx_name=saved_name)
        compare_onnx_pth(torch_net=torch_net, input_shape=input_shape, onnx_name=saved_name)
    else:
        print("skip onnx check","red")