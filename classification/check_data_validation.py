import os,sys
import numpy as np
from utils.auxdata import load_aux_data
import cv2

def delete_image_and_annotation(image_path):
    print("!!!delete image/annotation: ",image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
    anno_path = image_path + ".json"
    if os.path.exists(anno_path):
        os.remove(anno_path)
    return
        
def check_validation(indir, require_annotation=True, verbose=False):
    for rdir, _, names in os.walk(indir):
        for fn in names:
            ext = os.path.splitext(fn)[-1]
            if ext.lower() not in {'.jpg', '.bmp', '.png','.jpeg'}:
                continue
            fn = os.path.join(rdir,fn)
            if verbose:
                print("check data: ",fn)
            try:
                with open(fn, 'rb') as f:
                    raw = f.read()
                    if len(raw) == 0:
                        print("!!!!!empty image: ",fn)
                        delete_image_and_annotation(fn)
                        continue
                image_data = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), 1)
                if image_data is None:
                    print("!!!!!bad image: ",fn)
                    delete_image_and_annotation(fn)
                    continue
                image_tags, instances = [], []
                if os.path.exists(fn + ".json"):
                    image_tags, instances = load_aux_data(fn)
                if require_annotation and image_tags == [] and instances == []:
                    delete_image_and_annotation(fn)
                    raise Exception("miss json file for image ",fn)
            except:
                delete_image_and_annotation(fn)
                print("!!!!!ERROR data: ",fn)
                continue
    return 

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-data",help="input image+json file",required=True)
    ap.add_argument("--verbose",help="verbose",action="store_true")
    args = ap.parse_args()
    check_validation(args.data,require_annotation=False,verbose=args.verbose)
        


