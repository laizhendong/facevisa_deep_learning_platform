
import logging
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys

if sys.platform == "win32":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    print("PL_TORCH_DISTRIBUTED_BACKEND:",os.environ["PL_TORCH_DISTRIBUTED_BACKEND"])
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import datetime

from config.default_config import get_default_config

from eval import eval_net
from symbol.resnet import RESNET
from symbol.wideresnet import WRESNET

from utils.sampler import BalancedSampler

from data.default_dataset import CLASSICAL_CLASSIFIER_BASELINE,BasicDatasetLMDB
from torch.utils.data import DataLoader
from utils.torch_utils import select_device,set_seed
from train import train_net, setup_logging,UpdateOutputFolder,save_model_dist
from accelerate import Accelerator

WORK_DIR = os.getcwd()

def main(cfg,train_list, val_list, output_dir):
    accelerator = Accelerator()
    accelerator.init_trackers(
        project_name="classification",
        config={
            "learning_rate":cfg.SOLVER.LR_BASE,
            "epochs":cfg.SOLVER.EPOCHS
        }
    )
    
    setup_logging(os.path.join(WORK_DIR, output_dir))
    if args.randoff:
        set_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark= False
        torch.backends.cudnn.enabled = False
        logging.info("random OFF")
    #else:
    #    torch.backends.cudnn.deterministic = False
    #    torch.backends.cudnn.benchmark= True
    #    torch.backends.cudnn.enabled = False # set true will lower GPU usage
    #    logging_info("random ON")

    
    if accelerator:
        device = accelerator.device
    else:
        device = select_device(cfg.SOLVER.DEVICE)
    logging.info(f'Using device {cfg.SOLVER.DEVICE}')

    if cfg.DATA.NAME == "classical_classifier":
        dataset_train = CLASSICAL_CLASSIFIER_BASELINE(list_file=train_list, train_flag=True, cfg=cfg)
        dataset_test = CLASSICAL_CLASSIFIER_BASELINE(list_file=val_list, train_flag=False, cfg=cfg)
    elif cfg.DATA.NAME == "classical_classifier_lmdb":
        dataset_train = BasicDatasetLMDB(cfg,train_list, train_flag=True)
        dataset_test = BasicDatasetLMDB(cfg,val_list, train_flag=False)
    else:
        logging.error(f"unk dataset {cfg.DATA.NAME}")
        sys.exit(0)

    class_num_each_output = []
    for out_idx in range(dataset_train.output_num()):
        class_num_each_output.append(dataset_train.class_num(out_idx))
        
    if cfg.MODEL.BACKBONE_NAME.lower().startswith("resnet"):
        net = RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME,
                     pretrained_backbone = cfg.MODEL.USE_IMAGENET_BACKBONE,
                     input_channels=dataset_train.input_channels(),
                     num_classes=class_num_each_output)
    elif cfg.MODEL.BACKBONE_NAME.lower().startswith("wideresnet"):
        net = WRESNET(cfg.MODEL.BACKBONE_NAME,num_classes=dataset_train.class_num())
    else:
        logging.error(f"unk model {cfg.MODEL.BACKBONE_NAME}")
        sys.exit(0)
    logging.info(f'Network:\n'
                 f'\t{class_num_each_output} output channels (classes)\n')

    if cfg.MODEL.WEIGHTS != "":
        weights = torch.load(cfg.MODEL.WEIGHTS)
        net.load_state_dict(weights,strict=False)

    net.to(device=device)

    try:
        train_net(cfg,net,dataset_train, dataset_test,device,accelerator=accelerator)
    except KeyboardInterrupt:
        save_model_dist("INTERRUPTED.pth",net,accelerator)
        logging.info('Saved interrupt')
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_file",help="yaml configuration file")
    ap.add_argument("--randoff",help="do sth to make training result reproduced",action="store_true",default=False)
    args = ap.parse_args()
    cfg = get_default_config()
    cfg.merge_from_file(args.yaml_file)
    UpdateOutputFolder(cfg)
    cfg.freeze()
    main(cfg)
