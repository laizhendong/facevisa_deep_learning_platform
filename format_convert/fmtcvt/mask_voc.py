#########################################
#参考voc语义分割的格式，额外增加一个json文件，把像素值映射到标签名
#########################################
import cv2
import numpy as np
from .imgdesc import OBJDESC,IMGDESC
import json
import traceback
import matplotlib.pyplot as plt

def _encode(mask, bgvalue=None):
    data = mask.flatten()
    startp,startv,lens  = [0], [data[0]],[]
    curlen = 1
    for p in range(1,len(data)):
        if data[p] == startv[-1]:
            curlen += 1
        else:
            lens.append(curlen)
            startp.append(p)
            startv.append(data[p])
            curlen = 1
    lens.append(curlen)
    rle = []
    if bgvalue is not None:
        for p,l,v in zip(startp, lens,startv):
            if v != bgvalue:
                rle.extend([p,l])
    else:
        for p, l in zip(startp, lens):
            rle.extend([p,l])
    return rle

def _decode(rle,shape,fgvalueRGB = (255,255,255), bgvalue=(0,0,0), mask_init = None):
    H,W = shape
    assert isinstance(bgvalue,(tuple,list)) and np.all(fgvalueRGB) >= 0 and np.all(fgvalueRGB) <= 255
    assert isinstance(fgvalueRGB,(tuple,list)) and np.all(bgvalue) >= 0 and np.all(bgvalue) <= 255
    if mask_init is not None:
        mask = mask_init
    else:
        mask = np.zeros((H,W,3),dtype=np.uint8) + np.reshape(np.array(bgvalue,dtype=np.uint8),(1,1,3))
    maskB = mask[:,:,0].flatten()
    maskG = mask[:,:,1].flatten()
    maskR = mask[:,:,2].flatten()
    for p, l in zip(rle[0::2], rle[1::2]):
        maskR[p:p+l] = fgvalueRGB[0]
        maskG[p:p+l] = fgvalueRGB[1]
        maskB[p:p+l] = fgvalueRGB[2]
    maskR = np.reshape(maskR,order="C",newshape=(H,W,1)).astype(np.uint8)
    maskG = np.reshape(maskG,order="C",newshape=(H,W,1)).astype(np.uint8)
    maskB = np.reshape(maskB,order="C",newshape=(H,W,1)).astype(np.uint8)
    return np.concatenate([maskB,maskG,maskR],axis=-1)

def parse_name2rgb(config):
    rgb2name = {}
    try:
        name2rgb = config["name2rgb"]
        for name in name2rgb:
            rgb = name2rgb[name]
            rgb = "{},{},{}".format(rgb[0],rgb[1],rgb[2])
            rgb2name[rgb] = name
    except Exception as e:
        traceback.print_exc()
    return rgb2name


class SEGMENT_VOC(object):
    def __init__(self,names_to_export = None) -> None:
        self.name2rgb = None
        if names_to_export is not None:
            labelnames = list(set(names_to_export))
            cmap = plt.cm.get_cmap("jet")
            self.name2rgb = {}
            N = len(labelnames)
            for n,label in enumerate(labelnames):
                r,g,b,_ = cmap(float(n)/N)
                self.name2rgb[label] = (int(r * 255), int(g * 255), int(b * 255) )
             
        return 
    def Load(self,data_dict):
        mask_data_raw = data_dict['mask']
        rgb2name = parse_name2rgb(json.loads(data_dict['config']))
        imgdesc = IMGDESC()

        mask_data = np.fromstring(mask_data_raw,np.uint8)
        mask_data = cv2.imdecode(mask_data,flags=cv2.IMREAD_COLOR)
        if len(mask_data) == 2:
            H,W = mask_data.shape
        else:
            H,W,_ = mask_data.shape
        imgdesc.set("",W,H,3)
        
        mask_data = cv2.cvtColor(mask_data,cv2.COLOR_BGR2RGB) #BGR2RGB!!!!
        colors = np.array(list(set([(r,g,b) for (r,g,b) in mask_data.reshape(-1,3)]) - set([(0,0,0)])))
        colors = np.reshape(colors,(len(colors),1,1,3))
        mask_data_class = np.expand_dims(mask_data,0)
        mask_data_class = (mask_data_class == colors).astype(np.int64)
        mask_data_class = (np.sum(mask_data_class,axis=-1) == 3).astype(np.uint8) * 255
        for c in range(mask_data_class.shape[0]):
            rle = _encode(mask_data_class[c],bgvalue=0)
            rgb = colors[c,:,:,:].flatten().tolist()
            rgb = "{},{},{}".format(rgb[0],rgb[1],rgb[2])
            if rgb not in rgb2name.keys():
                print(f"skip color RGB: {rgb}")
                continue
            name = rgb2name[rgb]
            objdesc = OBJDESC()
            objdesc.set(name,-1,"Rle",rle,1.0)
            imgdesc.add_object(objdesc)
            if 0:
                rle_decode = _decode(rle,mask_data_class[c].shape)[:,:,0]
                err = np.all(mask_data_class[c] == rle_decode)
                if not err:
                    print("different after recovered")
                cv2.imshow(f"class {name} src",mask_data_class[c]) 
                cv2.imshow(f"class {name} rec",rle_decode) 
                cv2.imshow(f"class {name} err",(mask_data_class[c] == rle_decode).astype(np.uint8) * 255) 
                cv2.waitKey(-1)
        return imgdesc

    def Save(self,imgdesc):
        assert(self.name2rgb is not None),"names_to_export should be not None"
        W,H = imgdesc.width, imgdesc.height 
        mask = None
        for obj in imgdesc.objects:
            name = obj.names[0]
            shape_type = obj.shape_name
            if shape_type.lower() != "rle":
                continue #ignore others
            assert(name in self.name2rgb.keys()), f"{name} not in name2rgb.keys()"
            r,g,b = self.name2rgb[name]
            mask = _decode(obj.shape_data,(H,W),(r,g,b),(0,0,0),mask)
        _,encoded_mask = cv2.imencode(".png",mask)
        return encoded_mask
    def GetConfig(self):
        #assert(self.name2rgb is not None),"names_to_export should be not None"
        if self.name2rgb is None:
            return None
        return json.dumps(self.name2rgb,ensure_ascii=False,indent=4)
        


if __name__ == "__main__":
    
    with open("n:/2007_001311.png","rb") as f:
        mask_data = f.read()
    with open("n:/segmentation_name2rgb.json",'r',encoding='utf-8') as f:
        config = f.read()
    seg = SEGMENT_VOC()
    imgdesc = seg.Load({"mask":mask_data, "config":config})
    
    


