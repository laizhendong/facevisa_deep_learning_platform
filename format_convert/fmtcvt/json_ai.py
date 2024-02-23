import json
import warnings
import traceback
from .imgdesc import IMGDESC, OBJDESC

_LOADERS = {}
def add_loader(name):
    def _add_loader(value):
        if not callable(value):
            raise Exception(f"must be callable {value}")
        if name in _LOADERS.keys():
            raise Exception(f"{name} already registed")
        _LOADERS[name] = value
        return value
    return _add_loader

def CreateInstanceRect(object):
    try:
        w = object.shape_data[2] - object.shape_data[0] + 1
        h = object.shape_data[3] - object.shape_data[1] + 1
        data = {
            "type":"Rect",
            "top":object.shape_data[1],"left":object.shape_data[0],"width":w,"height":h,
            "attributes": [
                {
                    "class_id":object.ids[0],
                    "class_name": object.names[0] 
                }
            ]
        }
    except:
        traceback.print_exc()
    return data

@add_loader("Rect")
def LoadInstanceRect(json_data):
    try:
        x0,y0, w, h =  int(json_data['left']),int(json_data['top']),int(json_data['width']),int(json_data['height'])
        x1,y1 = x0 + w - 1, y0 + h - 1
        #print(json_data['attributes'],json_data)
        class_id = int(json_data['attributes'][0]['class_id']) 
        classname= json_data["attributes"][0]['class_name']
    except:
        traceback.print_exc()
    obj = OBJDESC()
    obj.set(classname,class_id, "rect", (x0,y0,x1,y1))
    return obj
    

#oou
def CreateInstancePoints(object):
    try:
        w = object.shape_data[2] - object.shape_data[0] + 1
        h = object.shape_data[3] - object.shape_data[1] + 1
        data = {
            "attributes":
                [{
                    "class_name":object.names[0],
                    "class_id":object.ids[0],
                }],
            "type": "Points",
            "points": [],
            "bbox":{"left":0,"right":0,"top":0,"bottom":0},
        }
        xyall = [[int(x),int(y)] for (x,y) in zip(object.shape_data[0::2], object.shape_data[1::2])]
        data['points'] = xyall
        Xall,Yall = [xy[0] for xy in xyall], [xy[1] for xy in xyall]
        data['bbox']['left'] = int(min(Xall))
        data['bbox']['right'] = int(max(Xall))
        data['bbox']['top'] = int(min(Yall))
        data['bbox']['bottom'] = int(max(Yall))         
    except:
        traceback.print_exc()
    return data

#oou
@add_loader("Points")
def LoadInstancePoints(json_data):
    try:
        class_id = int(json_data['attributes'][0]['class_id']) 
        classname= json_data["attributes"][0]['class_name']
        xylist = []
        for (x,y) in json_data['points']:
            xylist.extend((x,y))
    except:
        traceback.print_exc()
    obj = OBJDESC()
    obj.set(classname,class_id, "points", xylist)
    return obj

def CreateInstanceTemplate(object):
    try:
        points, pointLabels = [], {}
        for index, (x,y) in enumerate(zip(object.shape_data[0::2],object.shape_data[1::2])):
            points.append( {"id":f"{index}", "x":x, "y":y} )
            pointLabels[f"{index}"] = f"{index + 1}"
        data = {
            "type": "Template",
            "points": points,
            "pointLabels":pointLabels,
        }
    except:
        traceback.print_exc()
    return data

@add_loader("Template")
def LoadInstanceTemplate(json_data):
    try:
        points = json_data['points']
        pointsLabels = {} 
        pointsLabels_str = json_data['pointLabels']
        for idx in pointsLabels_str.keys():
            val = pointsLabels_str[idx]
            pointsLabels[int(idx)] = int(val)
        points = sorted( points, key = lambda p: pointsLabels[int(p['id'])], reverse=False)
        xy = []
        for pt in points:
            xy.extend( [pt['x'], pt['y']] )
        classname = ""
    except:
        traceback.print_exc()
    obj = OBJDESC()
    obj.set(classname, -1, "pts", xy, 1.0)
    return obj

    

def CreateInstancePolygon(object):
    try:
        inst = {
            "attributes":
                [{
                    "class_id": object.ids[0],
                    "class_name":object.names[0] 
                }],
            "type": "Polygon",
            "points": [],
            "left":0,"top":0,
            "pointsOffset":{"x":0,"y":0}
        }
        def _lines2polygon(lines):
            Xall, Yall = [float(x) for x in lines[0::2]],[float(y) for y in lines[1::2]]
            left,top = min(Xall), min(Yall)
            right,bottom = max(Xall), max(Yall)
            anchorX, anchorY = (left+right)/2.0,(top+bottom)/2.0
            pointsOffset = [anchorX - Xall[0], anchorY - Yall[0]]
            points = [[0,0]]
            for (x,y) in zip(Xall[1:], Yall[1:]):
                nx = round(x-anchorX+pointsOffset[0])
                ny = round(y-anchorY+pointsOffset[1])
                points.append((nx,ny))
            return points, pointsOffset , left, top
        points,pointsOffset, left, top = _lines2polygon(object.shape_data)
        inst['left'],inst['top'] = round(left),round(top)
        inst['pointsOffset']['x'] = round(pointsOffset[0])
        inst['pointsOffset']['y'] = round(pointsOffset[1])
        for (x,y) in points:
            inst['points'].extend([x,y])
    except:
        traceback.print_exc()
    return inst


@add_loader("Polygon")
def LoadInstancePolygon(json_data):
    try:
        def _polygon2lines(polygon_points,xy,offset):
            left,top = xy
            poly_xall = [x for x in polygon_points[0::2]]
            poly_yall = [y for y in polygon_points[1::2]]
            x0,x1 = min(poly_xall), max(poly_xall)
            y0,y1 = min(poly_yall), max(poly_yall)
            w,h = x1 - x0, y1 - y0
            anchorX, anchorY = left + w * 0.5, top + h * 0.5
            lines = []
            for (nx,ny) in zip(poly_xall, poly_yall):
                x = round(nx + anchorX - offset[0])
                y = round(ny + anchorY - offset[1])
                lines.extend((x,y))
            return lines
        polygon_points = json_data['points']
        xy = (json_data['left'], json_data['top'])
        offset = [json_data['pointsOffset']['x'],json_data['pointsOffset']['y']]
        data = _polygon2lines(polygon_points,xy,offset)
        classname = json_data["attributes"][0]['class_name']
        class_id = json_data['attributes'][0]['class_id']
    except: 
        traceback.print_exc()
    obj = OBJDESC()
    obj.set(classname,class_id, "polygon", data)
    return obj



def CreateInstanceRle(object):
    try:
        assert(object.shape_name.lower() == "rle"),"RLE input required"
        assert(isinstance(object.shape_data[0],int)), "RLE-ROW-FIRST"
        inst = {
            "attributes":
                [{
                    "class_id": object.ids[0],
                    "class_name":object.names[0] 
                }],
            "type": "Rle",
            "points": object.shape_data,
        }
    except:
        traceback.print_exc()
    return inst


@add_loader("Rle")
def LoadInstanceRle(json_data):
    try:
        data = json_data['points']
        classname = json_data["attributes"][0]['class_name']
        class_id = json_data['attributes'][0]['class_id']
    except: 
        traceback.print_exc()
    obj = OBJDESC()
    obj.set(classname,class_id, "rle", data)
    return obj
       
       
 
        
def LoadImageTags(json_data):
    tags = []
    if "image_tags" not in json_data.keys():
        return tags
    try:
        for image_tag in json_data['image_tags']:
            class_id = int(image_tag['class_id'])
            class_name = image_tag['class_name']
            tags.append({"class":class_name,"id":class_id})
    except:
        traceback.print_exc()
    return tags
def CreateImageTags(imagedes):
    tags = []
    if imagedes.classes == []:
        return tags
    try:
        for class_info in imagedes.classes:
            tags.append(
                {"class_name":class_info['class'],"class_id":class_info['id']}
            )
    except:
        traceback.print_exc()
    return tags
     
def CreateMetadata(width,height,name=""):
    try:
        data = {
                    "width":width,
                    "height":height,
                    "name":name,
        }
    except:
        traceback.print_exc()
    return data



def LoadMetadata(json_data):
    try:
        data = {
            "width":json_data['width'],
            "height":json_data['height'],
            "name":json_data['name'],
        }
    except:
        traceback.print_exc()
    return data    
    

def Load(json_str):
    json_data = json.loads(json_str)
    metadata = LoadMetadata(json_data['metadata'])
    imgdesc = IMGDESC()
    for inst_json in json_data['instances']:
        try:
            t = inst_json['type']
            imgdesc.objects.append(_LOADERS[t](inst_json))
        except:
            traceback.print_exc()
    imgdesc.add_classes( LoadImageTags(json_data) )
    imgdesc.set(metadata['name'],metadata['width'],metadata['height'],-1)
    return imgdesc

def Save(imgdesc): 
    objects = []
    for obj in imgdesc.objects:
        shape_name = obj.shape_name.lower()
        if shape_name == "rect":
            objects.append(CreateInstanceRect(obj))
        elif shape_name == "polygon":
            objects.append(CreateInstancePolygon(obj))
        elif shape_name == "points":
            objects.append(CreateInstancePoints(obj))
        elif shape_name == "pts":
            objects.append(CreateInstanceTemplate(obj))
        elif shape_name == "rle":
            objects.append(CreateInstanceRle(obj))
        else:
            assert(False),"unk shape name: {}".format(obj.shape_name)
            
    metadata = CreateMetadata(imgdesc.width, imgdesc.height,name = imgdesc.filename)
    image_tags = CreateImageTags(imgdesc)
    json_str = json.dumps({"metadata":metadata, "instances":objects,"image_tags":image_tags},ensure_ascii=False,indent=4)
    return json_str
     
if __name__ == "__main__":
    import cv2
    with open("n:/2.json",encoding="utf-8") as f:
        json_str = f.read()
    data = Load(json_str)
    img = cv2.imread('n:/2.jpg',1)
    xall = data.objects[0].shape_data[0::2]
    yall = data.objects[0].shape_data[1::2]
    xy = [(int(x),int(y)) for (x, y) in zip(xall,yall)] 
    for p0, p1 in zip(xy[0:-1],xy[1:]):
        cv2.line(img,p0,p1,(0,0,255),1)
    cv2.imwrite("n:/2a.jpg",img)
    json_str = Save(data)
    with open("n:/2a.json","w",encoding='utf-8') as f:
        f.write(json_str)