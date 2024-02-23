import numpy as np
from .imgdesc import IMGDESC,OBJDESC

def Load(pts_str):
    pts_str = pts_str.replace("\r",'').strip()
    lines = pts_str.split('\n')
    total = int(lines[1].split(':')[-1].strip())
    head_size = 3
    pts = []
    for offset in range(total):
        line = lines[head_size+offset].strip()
        line = line.replace('\t',' ')
        xy = [int(float(x)) for x in line.split(' ')]
        pts.extend(xy)
    imgdesc = IMGDESC()
    imgdesc.set("",0,0,0)
    obj = OBJDESC()
    obj.set("",-1,"pts",pts)
    imgdesc.add_object(obj)
    return imgdesc

def Save(imgdesc):
    xy = imgdesc.objects[0].shape_data
    lines = ["version:1"]
    xall,yall = [int(float(x)) for x in xy[0::2]], [int(float(y)) for y in xy[1::2]]
    lines.append('n_points:{}'.format(len(xall)))
    lines.append("{")
    for (x,y) in zip(xall,yall):
        lines.append(f"{x} {y}")
    lines.append("}")
    return '\n'.join(lines)


    