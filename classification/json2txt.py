
import os
import random
import argparse
from utils.auxdata import load_aux_data
#遍历train 和 test 路径下的json文件


def get_caffe_dataset(root,split,outdir):
    os.makedirs(outdir,exist_ok=True)
    paths, labels = [],[]
    for rdir, _, names in os.walk(root):
        for name in names:
            try:
                _,ext = os.path.splitext(name)
                if ext.lower() in {'.jpg','.jpeg','.bmp','.png'}:
                    image_path = os.path.join(rdir,name)
                    image_tags, instances = load_aux_data(image_path)
                    assert instances == [], "instances is not supported now!"
                    for image_tag in image_tags:
                        label = image_tag['class_id']
                        labels.append( label  )
                        paths.append(image_path)    
            except Exception as e:
                print("!!!!! ",e)
                continue 
    labels_unique = sorted(list(set(labels)))
    label2idx = {}
    idx2label_str = []
    for idx in range(len(labels_unique)):
        label = labels_unique[idx]
        label2idx[label] = idx
        idx2label_str.append(f"{idx} {label}")
    samples = []
    for path, label in zip(paths, labels):
        idx = label2idx[label]
        samples.append(f"{path} {idx}") 
    with open(os.path.join(outdir,f"{split}.txt"),"w") as f:
        f.write("\n".join(samples)) 
    random.shuffle(samples)
    with open(os.path.join(outdir,f"{split}_shuffle.txt"),"w") as f:
        f.write("\n".join(samples)) 
   
    with open(os.path.join(outdir,f"label_to_id_{split}.txt"),'w') as f:
        f.write('\n'.join(idx2label_str))
    return
    
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-data",help="data location",default="/data/")
    args = ap.parse_args()
    splits = [split for split in os.listdir(args.data) if split not in {".",".."} and os.path.isdir(os.path.join(args.data,split))]
    for split in splits:
        print("convert dataset: {}".format(os.path.join(args.data,split)))
        #print("output txt filename is set according to directory name")
        get_caffe_dataset(os.path.join(args.data,split), split, args.data)
    
        