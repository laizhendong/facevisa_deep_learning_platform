#目前仅支持单输出分支

import os,sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from symbol import resnet
from symbol import wideresnet
from utils import optional_file

from utils.modelcard import MODELCARD
from data import default_dataset
from collections import OrderedDict
from config.default_config import get_default_config
from symbol.repvgg import get_RepVGG_func_by_name, func_dict
from utils.auxdata import load_aux_data,save_aux_data
import warnings
import traceback
from utils import experiments
warnings.simplefilter('ignore')


       
def predict_img(net,image_data,device,transforms=None):
    H, W, _ = image_data.shape
    if not transforms is None:
        for augment in transforms:
            image_data = augment.forward(image_data)

    net.eval()

    image_data = np.transpose(image_data,(2,0,1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)
    image_data = image_data.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(image_data)
        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)
        probs = probs.squeeze().cpu().numpy()

    labels_data = np.argmax(probs,axis=0)
    probs_data = np.max(probs,axis=0)

    return probs_data, labels_data



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("exp_config",help="exp configuration file")
    ap.add_argument("-model_path",help="pth file path",required=True)
    ap.add_argument("-indir",help="input image dir",required=True)
    ap.add_argument("-outdir",help="output dir",required=True)
    args = ap.parse_args()
    exp_cfg = experiments.load_config(args.exp_config,'demo')
    cfg = get_default_config()
    cfg.merge_from_file(exp_cfg['hyparam_file'])
    cfg = optional_file.load_optional_file(cfg,exp_cfg['optional_file'])
    cfg.freeze()
    os.makedirs(args.outdir, exist_ok=True)


    print("start to load global classes...")
    global_classes =[]
    try:
        modelcard = MODELCARD(task='classification',version=1)
        with open(args.model_path + ".json",'r',encoding='utf-8') as f:
            json_data = f.read()
            modelcard.fromjson(json_data)
            modelcard.convert_name2global()
        print(modelcard.tojson())
        classes_local2global = []
        for l,g in modelcard.classes_local2global.items():
            classes_local2global.append((int(l),int(g)))
        classes_local2global = sorted(classes_local2global,key = lambda p : p[0],reverse=False)
        global_classes = [c[1] for c in classes_local2global ]
    except Exception as e:
        print(e)
        sys.exit(0)
    print("global_classes:",global_classes)
    assert(len(global_classes) > 0)
            
    # build net 
    backbone_name =  cfg.MODEL.BACKBONE_NAME
    if backbone_name.startswith('resnet'):
        net = resnet.RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME, input_channels = 3, num_classes=[len(global_classes)], pretrained_backbone=False)
        net.load_state_dict(torch.load(args.model_path), strict=True)
    elif backbone_name.lower().startswith("wideresnet"):
        net = wideresnet.WRESNET(cfg.MODEL.BACKBONE_NAME,num_classes=len(global_classes))
        state_dict = torch.load(args.model_path)['ema_model']
        state_dict_updated = OrderedDict()
        for name, value in  state_dict.items():
            if name.startswith("module."):
                nm = '.'.join(name.split('.')[1:])
                state_dict_updated[nm] = value
            else:
                state_dict_updated[name] = value
        net.load_state_dict(state_dict_updated, strict=True)
    elif backbone_name.startswith('RepVGG'):
        repvgg_model_list = list(func_dict.keys())
        if not backbone_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, backbone_name))

        repvgg_build_func = get_RepVGG_func_by_name(backbone_name)
        net = repvgg_build_func(deploy=True, num_classes=len(global_classes))
        net.load_state_dict(torch.load(args.model_path), strict=True)


    print("{}: {:.2f}M".format(backbone_name.lower(), sum(p.nelement() for p in net.parameters())/1e6))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    transforms = default_dataset.ClassificationDataset(cfg=cfg,list_file=None,train_flag=False).val_transforms

    preds = []
    for fn in os.listdir(args.indir):
        try:
            ext = os.path.splitext(fn)[-1]
            if ext.lower() not in {'.jpg', '.bmp', '.png','.jpeg'}:
                continue
            fn = os.path.join(args.indir,fn)
            image_data = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), 1)
            H,W,_ = image_data.shape
            image_tags, instances = load_aux_data(fn)
            if image_tags == [] and instances == []:
                image_tags = [
                    {"class_id":-1, "class_name":"", "class_score":"0.0"}
                ]
            for kk in range(len(image_tags)): #todos: speedup
                roi_data = image_data
                probs_data, labels_data = predict_img(net=net,image_data=roi_data,transforms=transforms,device=device)
                image_tags[kk]['class_id'] = global_classes[int(labels_data)]
                image_tags[kk]['class_name'] = ""
                image_tags[kk]['class_score'] = "{:.3f}".format(probs_data) 
            for kk in range(len(instances)): #todos: speedup
                x,y,w,h = instances[kk]['xywh']
                if x + w > W: 
                    w = W - x
                if y + h > H:
                    h = H - y
                roi_data = image_data[y:y+h,x:x+w]
                probs_data, labels_data = predict_img(net=net,image_data=roi_data,transforms=transforms,device=device)
                instances[kk]['class_id'] = global_classes[int(labels_data)]
                instances[kk]['class_name'] = "" 
                instances[kk]['class_score'] = "{:.3f}".format(probs_data) 
            save_aux_data(os.path.join(args.outdir,os.path.basename(fn)+".json"), image_tags, instances, W, H)
        except:
            print("!!!!!!!!sth goes wrong. result may be incorrect")
            traceback.print_exc()            

                               


