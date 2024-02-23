# by now only support one output branch
import os
import torch
import torch.nn.functional as F
from utils.metrics import MetricsAll
import argparse
from utils.modelcard import MODELCARD
from config.default_config import get_default_config
from data.default_dataset import CLASSICAL_CLASSIFIER_BASELINE
from utils.torch_utils import select_device
from torch.utils.data import DataLoader
from symbol import resnet
from symbol.repvgg import get_RepVGG_func_by_name, func_dict, repvgg_model_convert
import json
from utils import optional_file
from utils import experiments

def eval_net(net, loader, device,class_num_list):
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    assert(len(class_num_list) == 1) #!!!!
    MetricsObjects = [] 
    for class_num in class_num_list:
        MetricsObjects.append( MetricsAll(class_num,name="Validset") )

    for batch in loader:
        imgs, true_masks = batch['image'], batch['label']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_labels = true_masks.to(device=device, dtype=torch.long)
        with torch.no_grad():
            labels_pred = net(imgs)

        if len(class_num_list) > 1:
            for out_idx, label_pred in enumerate(labels_pred):
                tot += F.cross_entropy(label_pred, true_labels[:,out_idx].long()).item()
                MetricsObjects[out_idx].update(true_labels[:,out_idx].cpu(), label_pred.cpu())
        else:
            tot += F.cross_entropy(labels_pred, true_labels[:,0].long()).item()
            MetricsObjects[0].update(true_labels[:,0].cpu(), labels_pred.cpu())

    return {"loss":tot/(n_val if n_val > 0 else 0.000001), "metrics":MetricsObjects}


if __name__ == "__main__":


    ap = argparse.ArgumentParser()
    ap.add_argument("exp_config",help="exp configuration file")
    ap.add_argument("-model_path",help="model path to eval",required=True)
    ap.add_argument("-listfile",help="list file",required=True)
    ap.add_argument("-outdir",help="outdir",default="./output")
    args = ap.parse_args()
    exp_cfg = experiments.load_config(args.exp_config,'eval')
    batch_size = exp_cfg["batch_size"]
    num_worker = exp_cfg["num_worker"]
    os.makedirs(args.outdir,exist_ok=True)

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
    
    
    num_classes =  len(global_classes)
    
    # read config
    cfg = get_default_config()
    cfg.merge_from_file(exp_cfg['hyparam_file'])
    cfg = optional_file.load_optional_file(cfg,exp_cfg['optional_file'])
    cfg.freeze()



    # construct the dataset 
    print("input list file",args.listfile)
    val_dataset = CLASSICAL_CLASSIFIER_BASELINE(list_file=args.listfile, train_flag=False, cfg=cfg)
    assert len(val_dataset) > 0 and val_dataset.class_num() > 0, "evaluation should run with labeled samples"
    val_loader = DataLoader(val_dataset,batch_size=batch_size,
                        shuffle=False, num_workers = num_worker, pin_memory=True, drop_last=False)


    # build net
    backbone_name =  cfg.MODEL.BACKBONE_NAME
    if backbone_name.startswith('resnet'):
        net = resnet.RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME, input_channels = 3, num_classes=[num_classes], pretrained_backbone=False)
    elif backbone_name.startswith('RepVGG'):
        repvgg_model_list = list(func_dict.keys())
        if not backbone_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, backbone_name))

        repvgg_build_func = get_RepVGG_func_by_name(backbone_name)
        net = repvgg_build_func(deploy=False, num_classes=val_dataset.class_num(), with_softmax=True) # val_dataset.class_num()

    device = select_device(cfg.SOLVER.DEVICE)

    net.load_state_dict(torch.load(args.model_path))
    net = net.to(device=device)


    thresholds_to_eval = [a / 100 for a in range(0,101)]
    val_info = eval_net(net, val_loader, device, [num_classes])

    rets = {
        "evaluation_Results": {
        "val_loss":"{:.5f}".format(val_info['loss']),
        "f1-score": "{:.3f}".format(val_info['metrics'][0].get('f1score')[1]),
        "recalling": "{:.3f}".format(val_info['metrics'][0].get('recalling')[1]),
        "precision": "{:.3f}".format(val_info['metrics'][0].get('precision')[1]),
        "accuracy": "{:.3f}".format(val_info['metrics'][0].get('accuracy')[1]),
        "detail": {}
        },
        "visual_metrics":[]
    }
    report = val_info['metrics'][0].get('classification_report_dict')[1]
    
    for local_class,global_class in enumerate(global_classes):
        local_class_str = f'{local_class}'
        if local_class_str in report.keys():
            if isinstance(report[local_class_str],dict):
                for k in report[local_class_str].keys():
                    if k == "support":
                        report[local_class_str][k] = int(report[local_class_str][k]) #be back-compatible
                    else:
                        report[local_class_str][k] = "{:.3f}".format(report[local_class_str][k])
            rets["evaluation_Results"]['detail'][global_class] = report[local_class_str]

    visual_metrics = val_info['metrics'][0].get_visual_metrics_with_confidence(thresholds_to_eval)

                    
    global_classes_with_neg = dict([(i+1,c) for i,c in enumerate(global_classes)])
    global_classes_with_neg[0] = -1
    cm_size = len(global_classes_with_neg.keys())
    for th, visual_metric in zip(thresholds_to_eval, visual_metrics):
        cm = visual_metric["cm"]
        rec_data = []
        confusion_data = []
        for row in range(1,cm_size):
            label_gt = global_classes_with_neg[row]
            for col in range(cm_size):
                label_pred = global_classes_with_neg[col]
                n = int(cm[row,col])
                data = [f"{label_gt}",f"{label_pred}",n]
                if n == 0:
                    continue
                if row == col:
                    rec_data.append(data)
                else:
                    confusion_data.append(data)
        rets["visual_metrics"].append(
            {
                "thresh": f"{th:.2f}",
                "confusion_on_diag":rec_data,
                "confusion_off_diag":confusion_data,
                "precision":"{:.3f}".format(visual_metric["precision"]),
                "accuracy":"{:.3f}".format(visual_metric['accuracy']),
                "recall":"{:.3f}".format(visual_metric['recall'])
            }
        )       
    with open(os.path.join(args.outdir,"results.json"),'w') as f:
        json.dump(rets,f,indent=4)
            
    # eval converted net
    if backbone_name.startswith('RepVGG'):
        converted_model_savepath = args.model_path.replace(".pth", "_converted.pth")
        deploy_repvgg = repvgg_model_convert(net, save_path=converted_model_savepath)
        print("\n\nrepvgg converted deploy eval")
        val_info = eval_net(deploy_repvgg, val_loader, device, val_dataset.class_num())
        print("save converted repvgg in {}".format(converted_model_savepath))
