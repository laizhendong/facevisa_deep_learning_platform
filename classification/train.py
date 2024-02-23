
import logging
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import argparse
import numpy as np
import torch,json
import torch.nn as nn
from torch import optim
import datetime
from utils import optional_file
from utils import experiments
from config.default_config import get_default_config

from eval import eval_net
from symbol.resnet import RESNET
from symbol.wideresnet import WRESNET
from symbol.alexnet import ALEXNET
from symbol.repvgg import *
from utils.sampler import BalancedSampler

from data.default_dataset import CLASSICAL_CLASSIFIER_BASELINE
from torch.utils.data import DataLoader
from utils.torch_utils import select_device,set_seed

from functools import partial

from utils.modelcard import MODELCARD

from torch.utils.tensorboard import SummaryWriter


def setup_logging(outdir):
    os.makedirs(outdir,exist_ok=True)
    log_name = os.path.splitext(os.path.basename(__file__))[0]
    logging.basicConfig(
        level=logging.DEBUG,
        format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename= os.path.join(outdir,log_name + ".log"),
        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return

def save_model_dist(path,net,accelerator):
    if accelerator:
        if accelerator.is_main_process:
            uwnet_states = accelerator.unwrap_model(net.state_dict())
            torch.save(uwnet_states, path)
    else:
        torch.save(net.state_dict(),path)
    return
    
def logging_info_dist(args, accelerator):
    if accelerator:
        if accelerator.is_main_process:
            logging.info(args)
    else:
        logging.info(args)
    return
        
        

def train_net(cfg,net,dataset_train, dataset_test,device,final_model_path,accelerator=None,summary_writer=None):

    logging_info = partial(logging_info_dist,accelerator=accelerator)
    save_model = partial(save_model_dist,accelerator=accelerator)    

    class_num_list = [dataset_train.class_num(o) for o in range(dataset_train.output_num())]

    top_f1score_val = 0

    shuffle_flag = True
    custom_sampler = None
    if cfg.DATA.CLASSES_BALANCED:
        assert(len(class_num_list) == 1), "only one-output supports balanced sampling"
        custom_sampler = BalancedSampler(dataset_train.get_index_wrt_label())
        shuffle_flag = False
    train_loader = DataLoader(dataset_train, batch_size=cfg.SOLVER.BATCH_SIZE,
                              shuffle=shuffle_flag,
                              num_workers=cfg.DATA.THREAD_NUM, pin_memory=True,sampler=custom_sampler,
                              prefetch_factor=cfg.DATA.PREFETCH_FACTOR)
    val_loader = DataLoader(dataset_test,batch_size=cfg.SOLVER.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.DATA.THREAD_NUM, pin_memory=True, drop_last=False)
    global_step = 0
    logging_info("------------------------------------")
    logging_info(f"{cfg}")
    logging_info("------------------------------------")


    optimizer = optim.SGD(net.parameters(), lr=cfg.SOLVER.LR_BASE, weight_decay=5e-5, momentum=0.9)

    if cfg.SOLVER.LR_POLICY == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    elif cfg.SOLVER.LR_POLICY == 'cosine':
        max_iter = (cfg.SOLVER.EPOCHS * len(dataset_train) + cfg.SOLVER.BATCH_SIZE - 1)//cfg.SOLVER.BATCH_SIZE
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_iter,eta_min=1e-9,last_epoch=-1)
    else:
        logging_info(f'unk lr policy: {cfg.SOLVER.POLICY}')
        sys.exit(0)

    if accelerator:
        net,optimizer,train_loader,val_loader,scheduler = accelerator.prepare(net,optimizer,train_loader,val_loader,scheduler)

    valid_run_freq = cfg.SOLVER.EPOCHS // 20
    valid_run_freq = 1 if valid_run_freq < 1 else valid_run_freq     
    valid_run_freq = 10 if valid_run_freq > 10 else valid_run_freq    

    criterion = nn.CrossEntropyLoss()
    t0_start = datetime.datetime.now()
    for epoch in range(cfg.SOLVER.EPOCHS):
        net.train()

        epoch_loss = 0
        epoch_batches = 0
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['label']


            imgs = imgs.to(device=device, dtype=torch.float32)
            true_labels = true_masks.to(device=device, dtype=torch.long)

            masks_pred = net(imgs)
            loss = 0
            if len(class_num_list) > 1:
                for out_idx,mask_pred in enumerate(masks_pred):
                    loss += criterion(mask_pred, true_labels[:,out_idx].long())
            else:
                loss += criterion(masks_pred,true_labels[:,0].long())
            epoch_loss += loss.item()
            epoch_batches += 1

            #pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
            if cfg.SOLVER.GRADIENT_MAX > 0:
                nn.utils.clip_grad_value_(net.parameters(), cfg.SOLVER.GRADIENT_MAX)
            optimizer.step()

            #pbar.update(imgs.shape[0])
            global_step += 1

            if cfg.SOLVER.LR_POLICY != 'plateau':
                scheduler.step()


            #here assume plateau winsize is 1024 iterations
            if global_step % 1024 == 0 and cfg.SOLVER.LR_POLICY == 'plateau':
                val_info = eval_net(net, val_loader, device, class_num_list)
                net.train()
                scheduler.step(val_info['loss'])

        if (epoch + 1) % valid_run_freq == 0 or epoch == 0:
            val_info = eval_net(net, val_loader, device, class_num_list)
            f1score_val = val_info['metrics'][0].get('f1score')[1] ##out-idx=0 by default
            if f1score_val > top_f1score_val:
                top_f1score_val = f1score_val
            logging_info(f"validate Loss {val_info['loss']:.3f}")
            for metric_name in ['f1score','confusion_matrix']:
                for out_idx in range(len(val_info['metrics'])):
                    name,val = val_info['metrics'][out_idx].get(metric_name)
                    logging_info(f'\t ==={out_idx} {name}===, \n {val}')
            if summary_writer:
                out_idx = 0
                summary_writer.add_scalar("avg_precision", val_info['metrics'][out_idx].get('precision')[1], epoch)
                summary_writer.add_scalar("avg_recall", val_info['metrics'][out_idx].get('recalling')[1], epoch)
                summary_writer.add_scalar("avg_f1-score", val_info['metrics'][out_idx].get('f1score')[1], epoch)
                summary_writer.add_scalar("accuracy", val_info['metrics'][out_idx].get('accuracy')[1], epoch)
                summary_writer.add_scalar("val_epoch_loss", val_info['loss'], epoch)
        time_train = (datetime.datetime.now() - t0_start).seconds/3600.0
               #tensorboard
        if summary_writer:
            summary_writer.add_scalar("train_epoch_loss", epoch_loss/epoch_batches, epoch)
        logging_info(f'Epoch {epoch+1} Loss {epoch_loss/epoch_batches:.3f} LR {scheduler.get_last_lr()[0]:.3e} TimeElapsed {time_train:.2f}h')

    if accelerator:
        accelerator.wait_for_everyone()
    save_model(final_model_path,net)
    logging_info(f'Checkpoint {epoch + 1} saved !')

    #make sure validation once at least
    val_info = eval_net(net, val_loader, device, class_num_list)
    f1score_val = val_info['metrics'][0].get('f1score')[1]
    if f1score_val > top_f1score_val:
        top_f1score_val = f1score_val

    return top_f1score_val


def load_id_and_name(label2id, id2name):
    local2global = {}
    global2name = {}    
    try:
        with open(label2id,mode='r',encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                local_class, global_class = [l.strip() for l in line.split(' ')]
                local2global[local_class] = global_class
        with open(id2name,mode='r',encoding='utf-8') as f:
            global2name = json.load(f)
    except Exception as e:
        print(e)
        sys.exit(0)    
    return local2global, global2name

def build_network(cfg):
    class_num_each_output = []
    for out_idx in range(dataset_train.output_num()):
        class_num_each_output.append(dataset_train.class_num(out_idx))

    if cfg.MODEL.BACKBONE_NAME.lower().startswith("resnet"):
        net = RESNET(backbone_name = cfg.MODEL.BACKBONE_NAME,
                     pretrained_backbone = cfg.MODEL.USE_IMAGENET_BACKBONE,
                     input_channels=dataset_train.input_channels(),
                     num_classes=class_num_each_output)
    elif cfg.MODEL.BACKBONE_NAME.lower().startswith("wideresnet"):
        assert(len(class_num_each_output) == 1) ##!!!!!!!!!todo
        net = WRESNET(cfg.MODEL.BACKBONE_NAME,num_classes=class_num_each_output[0])
    elif cfg.MODEL.BACKBONE_NAME.startswith("RepVGG"):
        assert(len(class_num_each_output) == 1) ##!!!!!!!!!todo
        model_name = cfg.MODEL.BACKBONE_NAME
        repvgg_model_list = list(func_dict.keys())
        if not model_name in repvgg_model_list :
            raise Exception("wrong name with repvgg , should be in {}, but get {}".format(repvgg_model_list, model_name))

        repvgg_build_func = get_RepVGG_func_by_name(model_name)
        net = repvgg_build_func(deploy=False, num_classes=dataset_train.class_num(0))  
    elif cfg.MODEL.BACKBONE_NAME.startswith("alexnet"):
        assert(len(class_num_each_output) == 1) ##!!!!!!!!!todo
        net = ALEXNET(backbone_name = cfg.MODEL.BACKBONE_NAME,
                    pretrained_backbone = cfg.MODEL.USE_IMAGENET_BACKBONE,
                    input_channels=dataset_train.input_channels(),
                    num_classes=dataset_train.class_num(0))        
    else:
        logging.error(f"unk model {cfg.MODEL.BACKBONE_NAME}")
        sys.exit(0)
    logging.info(f'Network:\n')
    for out_idx, class_num in enumerate(class_num_each_output):
        logging.info(f'\t{out_idx}: {class_num} output channels (classes)\n')     
    logging.info("{}: {:.2f}M".format(cfg.MODEL.BACKBONE_NAME.lower(), sum(p.nelement() for p in net.parameters())/1e6))        
    return net


    
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_conf",help="exp configuration file")
    ap.add_argument("-train_list",help="train txt",required=True)
    ap.add_argument("-val_list",help="val txt",required=True)
    ap.add_argument("-output_dir",help="fold to save log/model",required=True)
    ap.add_argument("-output_model_name",help="short name of last model",default="last.pth")
    ap.add_argument("-tensorboard_dir",help="tensorboard log dir",default="./output/exp0")
    args = ap.parse_args()
    exp_cfg = experiments.load_config(args.exp_conf,'train')
    cfg = get_default_config()
    
    cfg.merge_from_file(exp_cfg["hyparam_file"])
    cfg = optional_file.load_optional_file(cfg,exp_cfg['optional_file'])
    cfg.freeze()
    print("------------configuration after merge--------------")
    print(cfg)
    
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(args.tensorboard_dir,exist_ok=True)
    setup_logging(args.output_dir)
    
    
    dataset_train = CLASSICAL_CLASSIFIER_BASELINE(list_file=args.train_list, train_flag=True, cfg=cfg)
    dataset_test = CLASSICAL_CLASSIFIER_BASELINE(list_file=args.val_list, train_flag=False, cfg=cfg)
    device = select_device(cfg.SOLVER.DEVICE)
    logging.info(f'Using device {cfg.SOLVER.DEVICE}')



    print("start to load global classes...") 
    local2global,global2name = load_id_and_name(exp_cfg['label2id_file'], exp_cfg['id2name_file'])
    
    modelcard = MODELCARD(task='classification',version=1)        
    modelcard.set_classes(local2global,global2name=global2name)
    modelcard.set_name(cfg.MODEL.BACKBONE_NAME)
    modelcard.set_outputs([["prob"]])
    modelcard.set_mean_std(cfg.DATA.MEAN,cfg.DATA.STD)
    W,H = cfg.DATA.AUGMENT.RESIZE
    if np.prod(cfg.DATA.AUGMENT.CROP) > 0:
        W,H = cfg.DATA.AUGMENT.CROP        
    modelcard.set_inputs(
        [["data"]], [[-1,dataset_train.input_channels(),H,W]]
    )        
    
    
       
    net = build_network(cfg)
    if cfg.MODEL.WEIGHTS != "":
        weights = torch.load(cfg.MODEL.WEIGHTS)
        net.load_state_dict(weights,strict=False)
    net.to(device=device)        
    



    summary_writer = SummaryWriter(args.tensorboard_dir)
    try:
        finalmodel = os.path.join(args.output_dir,exp_cfg['output_model_name'])
        train_net(cfg,net,dataset_train, dataset_test,device,final_model_path=finalmodel, summary_writer=summary_writer)
        summary_writer.close()
        with open(finalmodel+".json","w",encoding='utf-8') as f:
            f.write(modelcard.tojson())
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')

