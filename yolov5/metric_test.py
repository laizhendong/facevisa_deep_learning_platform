import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    ConfusionMatrix, coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression, scale_coords,
    xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging)
from utils.torch_utils import select_device, time_synchronized
import re

curPath = os.path.dirname(os.path.realpath(__file__))
# 获取yaml文件路径
yamlPath = os.path.join(curPath, "alg_params.yaml")
 
# open方法打开直接读出来
f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
# 用load方法转字典
test_dict = yaml.load(cfg)  


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         metrics_json=False,
         nosave_results=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         plots=True):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels      
        if save_txt:
            #out = Path('/output')
            out = Path('/inference/output') # change
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(save_dir / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.2, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    
    #置信度从0.01-1的混淆矩阵计算
    #创建一个空列表来存储eval的每个子字典
    metrics_json = opt.metrics_json
    if metrics_json:
        rund = 101
    else:
        rund = 2
    eval_list = []
    for i in range(0, rund):  # 1到101，包括1不包括101
        conf_matrix = i * 0.01
        #matrix
        matrix,tp,fp,fn,tn = [[]], [], [], [], []
        confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_matrix, iou_thres=0.40)
        #matrix
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width
            whwh = torch.Tensor([width, height, width, height]).to(device)

            # Disable gradients
            with torch.no_grad():
                # Run model
                t = time_synchronized()
                inf_out, train_out = model(img, augment=augment)  # inference and training outputs
                t0 += time_synchronized() - t

                # Compute loss
                if training:  # if model has loss hyperparameters
                    loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

                # Run NMS
                t = time_synchronized()
                output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
                t1 += time_synchronized() - t

            # Statistics per image
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if pred is None:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                        #matrix
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        #matrix
                    continue

                # Append to text file
                if save_txt:
                    gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                    x = pred.clone()
                    x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                    for *xyxy, conf, cls in x:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(str(out / Path(paths[si]).stem) + '.txt', 'a') as f:
                            #f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                            #添加所属类别的置信度
                            f.write(('%g ' * 6 + '\n') % (cls, conf, *xywh))  # label format

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
                if save_json:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = Path(paths[si]).stem
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                      'category_id': coco91class[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                    
                    #matrix
                    predn = pred.clone()
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    confusion_matrix.process_batch(predn, labelsn)
                    matrix,tp,fp,fn,tn = confusion_matrix.tp_fp()
                    #matrix
                    
                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                        # Search for detections
                        if pi.shape[0]:
                            # Prediction to target ious
                            ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            # Plot images
            if plots and batch_i < 1:
                f = save_dir / ('test_batch%g_gt.jpg' % batch_i)  # filename
                plot_images(img, targets, paths, str(f), names)  # ground truth
                f = save_dir / ('test_batch%g_pred.jpg' % batch_i)
                plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions
        
        #matrix                    
        #计算精度和召回率
        epsilon = 1e-10       
        #if nc == 1:
            #accuracy = "{:.4f}".format((tp[0] + tn[0] + epsilon)/(tp[0] + fp[0] + tn[0] + fn[0] + epsilon))
            #recalling = "{:.4f}".format((tp[0] + epsilon)/(tp[0] + fn[0] + epsilon))
        
        if nc >= 1:
            accuracy = "{:.4f}".format((np.sum(tp) + np.sum(tn) + epsilon)/(np.sum(tp) + np.sum(fp) + np.sum(tn) + np.sum(fn) + epsilon))
            recalling = "{:.4f}".format((np.sum(tp) + epsilon)/(np.sum(tp) + np.sum(fn) + epsilon))
        
        # 初始化正检记录的列表
        ture_positives_list = []  
        for i in range(len(names)):
            ture_positives_list.append([names[i], names[i], int(tp[i])])
        
        # 获取类别数量
        names_add = names + ["-1"]
        num_classes = matrix.shape[0]

        # 初始化误检记录的列表
        false_positives_list = []
        # 遍历每个实际类别
        for actual_class_idx in range(num_classes):
            # 遍历每个预测类别
            for predicted_class_idx in range(num_classes):
                if actual_class_idx != predicted_class_idx:
                    # 该实际类别误检为预测类别的数量
                    #false_positives = matrix[actual_class_idx, predicted_class_idx]
                    false_positives = int(matrix[predicted_class_idx, actual_class_idx])
                    # 添加误检记录到列表中
                    false_positives_list.append([names_add[actual_class_idx], names_add[predicted_class_idx], false_positives])

        # 打印每个误检记录
        #for actual_class, predicted_class, false_positives in false_positives_list:
            #print(f"Actual Class {actual_class} misclassified as Predicted Class {predicted_class}: {false_positives} false positives")
         
        eval_info = {
            "thresh": "{:.2f}".format(conf_matrix),
            "confusion_on_diag": ture_positives_list,
            "confusion_off_diag": false_positives_list,
            "accuracy": accuracy,
            "recall": recalling,
        }
        
        # 将子字典添加到eval_list中
        eval_list.append(eval_info)                
          
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        
    nosave_results = opt.nosave_results
    # Print results
    pf = '%20s' + '%12.3f' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # creat test results.txt
    if nosave_results is False:   
        with open('/output/results.txt', 'a') as f:  
            #f.write(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95') + '\n')
            f.write(s + '\n')
            f.write(pf % ('all', seen, nt.sum(), mp, mr, map50, map) + '\n')
        
    # Print results per class
    if nosave_results is False:
        if verbose and nc > 0 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                # creat test results.txt
                with open('/output/results.txt', 'a') as f:                
                    f.write(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')
                
    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    
    # 读取result.txt文件
    result_file_path = '/output/results.txt'
    with open(result_file_path, 'r') as file:
        result_content = file.read()

    # 使用正则表达式提取评估指标数据
    pattern = r"(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    matches = re.findall(pattern, result_content)
    # 创建一个字典以保存评估指标数据
    evaluation_data = {
        "Class": [],
        "Images": [],
        "Targets": [],
        "P": [],
        "R": [],
        "mAP@0.5": [],
        "mAP@0.5:0.95": []
    }
    # 将提取的数据填充到字典中
    for match in matches:
        evaluation_data["Class"].append(match[0])
        evaluation_data["Images"].append(int(float(match[1])))
        evaluation_data["Targets"].append(int(float(match[2])))
        evaluation_data["P"].append(str(match[3]))
        evaluation_data["R"].append(str(match[4]))
        evaluation_data["mAP@0.5"].append(str(match[5]))
        evaluation_data["mAP@0.5:0.95"].append(str(match[6]))
    # 创建一个包含评估指标数据的字典
    data = {"evaluation_Results": evaluation_data,
            "visual_metrics": eval_list
            }
    # 指定要写入的 JSON 文件路径
    json_file_path = "/output/results.json"
    # 将数据写入 JSON 文件
    if metrics_json:
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)       
       
    # Save JSON
    if save_json and len(jdict):
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print('ERROR: pycocotools unable to run: %s' % e)

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data.yaml', help='*.data path')
    
    #parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    #parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    #parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
       
    parser.add_argument('--batch-size', type=int, default=test_dict['batch_size'], help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=max(test_dict['img_size']), help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=test_dict['conf_thres'], help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=test_dict['iou'], help='IOU threshold for NMS')
    
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--metrics-json', action='store_true', help='save metrics to *.json')
    parser.add_argument('--nosave-results', action='store_true', help='do not save metrics to results.txt')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             opt.metrics_json,
             opt.nosave_results)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot
