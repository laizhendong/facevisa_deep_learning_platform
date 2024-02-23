from xml.etree import ElementTree as ET 
import traceback
from xml.dom import minidom
from .imgdesc import OBJDESC,IMGDESC

"""
polygon of 4-edges will be convert to quad of anydraw
"""
SHAPE_TO_OBJ = { "rect":"rect", "pen":"points", "polygon":"polygon", "quad":"polygon"}
SHAPE_FROM_OBJ = { "rect":"rect", "points":"pen", "polygon":"polygon",'quad':'quad'} 


def prettify(xml_node):
    xml_content = ET.tostring(xml_node, 'utf-8')
    reparsed = minidom.parseString(xml_content)
    return reparsed.toprettyxml(indent="  ")

def _read_object_rect(shape):
    points = shape.findall('points')[0]
    xmin = int(points.findall('x')[0].text)
    ymin = int(points.findall('y')[0].text)
    xmax = int(points.findall('x')[1].text)
    ymax = int(points.findall('y')[1].text)
    return "rect",(xmin,ymin,xmax,ymax)

def _read_object_bezier(shape):
    points = shape.findall('points')[0]
    x0 = int(points.findall('x')[0].text)
    y0 = int(points.findall('y')[0].text)
    x1 = int(points.findall('x')[1].text)
    y1 = int(points.findall('y')[1].text)
    x2 = int(points.findall('x')[2].text)
    y2 = int(points.findall('y')[2].text)
    return "bezier", (x0,y0,x1,y1,x2,y2) 

def _read_object_pen(shape):
    points = shape.findall('points')[0]
    xlist, ylist = points.findall('x'), points.findall('y')
    points_list = []
    for x, y in zip(xlist, ylist):
        x,y = int(x.text), int(y.text)
        points_list.extend((x,y))
    return "pen",points_list

def _read_object_polygon(shape):
    points = shape.findall('points')[0]
    xlist, ylist = points.findall('x'), points.findall('y')
    xy_all = []
    for x, y in zip(xlist, ylist):
        x, y = int(x.text), int(y.text)
        xy_all.extend((x,y))
    return "polygon",xy_all


def _read_object_quad(shape):
    points = shape.findall('points')[0]
    xlist, ylist = points.findall('x'), points.findall('y')
    xy_all = []
    for x, y in zip(xlist, ylist):
        x, y = int(x.text), int(y.text)
        xy_all.extend((x,y))
    return "quad",xy_all

READ_OBJECT_CALLS = {}
READ_OBJECT_CALLS["rect"] = _read_object_rect
READ_OBJECT_CALLS["bezier"] = _read_object_bezier
READ_OBJECT_CALLS["pen"] = _read_object_pen
READ_OBJECT_CALLS["polygon"] = _read_object_polygon
READ_OBJECT_CALLS["quad"] = _read_object_quad

def LoadObject(xml_node):
    names = xml_node.findall('name')
    if len(names) == 0 or names[0].text == None:
        name = ''
    else:
        name = names[0].text.strip()
    levels = xml_node.findall('level')
    if len(levels) == 0 or levels[0].text == None:
        level = -1
    else:
        level = int(float((xml_node.findall('level')[0].text)))
    
    scores = xml_node.findall('score')
    if len(scores) > 0:
        score = float(scores[0].text)
    else:
        score = -1     
    shapes = xml_node.findall("shape")
    shape_type = ""
    if len(shapes) == 1:
        try:
            shape_type,xys = READ_OBJECT_CALLS[shapes[0].attrib["type"]](shapes[0])
        except Exception as e:
            traceback.print_exc()
    obj = OBJDESC()
    if shape_type in SHAPE_TO_OBJ.keys():
        shape_name = SHAPE_TO_OBJ[shape_type]
    else:
        raise Exception(f"unk shape type: {shape_type}")
    obj.set(name,-1,shape_name,xys,confidence=score)
    return obj
    


def _write_object_rect(parent_node,xylist):
    points = ET.SubElement(parent_node, "points")
    ET.SubElement(points, "x").text=str(int(xylist[0]))
    ET.SubElement(points, "y").text=str(int(xylist[1]))
    ET.SubElement(points, "x").text=str(int(xylist[2]))
    ET.SubElement(points, "y").text=str(int(xylist[3]))
    return


def _write_object_pen(parent_node,xylist):
    points = ET.SubElement(parent_node, "points")
    for (x,y) in zip(xylist[0::2],xylist[1::2]):
        ET.SubElement(points, "x").text = str(int(x))
        ET.SubElement(points, "y").text = str(int(y))
    return

def _write_object_quad(parent_node,xylist):
    points = ET.SubElement(parent_node, "points")
    for (x,y) in zip(xylist[0::2],xylist[1::2]):
        ET.SubElement(points, "x").text = str(int(x))
        ET.SubElement(points, "y").text = str(int(y))
    return

WRITE_OBJECT_CALLS = {}
WRITE_OBJECT_CALLS["rect"] = _write_object_rect
WRITE_OBJECT_CALLS["bezier"] = _write_object_pen
WRITE_OBJECT_CALLS["pen"] = _write_object_pen
WRITE_OBJECT_CALLS["polygon"] = _write_object_pen
WRITE_OBJECT_CALLS["quad"] = _write_object_quad

def AddObject(xml_node,object, level = 0):
    subnode=ET.SubElement(xml_node, "object")
    ET.SubElement(subnode, "name").text=object.names[0]
    ET.SubElement(subnode, "pose").text="Unspecified"
    ET.SubElement(subnode, "truncated").text='0'
    ET.SubElement(subnode, "difficult").text='0'
    ET.SubElement(subnode,'staintype').text = ""
    ET.SubElement(subnode,'area').text = "0.00"
    ET.SubElement(subnode,'level').text = '{}'.format(int(float(level)))
    ET.SubElement(subnode,'score').text = "{}".format(object.scores[0])
    shape = ET.SubElement(subnode,'shape')
    
    shape_type = ""
    shape_name = object.shape_name
    if shape_name in SHAPE_FROM_OBJ.keys():
        shape_type = SHAPE_FROM_OBJ[shape_name]
    else:
        raise Exception(f"unknown shape name {shape_name}")
    shape.attrib['type'] = shape_type
    shape.attrib["color"] = "Blue"
    shape.attrib['thickness'] = "3"
    WRITE_OBJECT_CALLS[shape_type](shape,object.shape_data) 
    return xml_node

def LoadHeader(xml_node):
    sizes = xml_node.findall('size')[0]
    width = int(float(sizes.findall('width')[0].text))
    height = int(float(sizes.findall('height')[0].text))
    depth = int(float(sizes.findall('depth')[0].text))
    if xml_node.findall("filename") != []:
        filename = xml_node.findall("filename")[0].text
    elif xml_node.findall("file_name") != []:
        filename = xml_node.findall("file_name")[0].text
    else:
        filename = ""
    return {
        "width":width, "height":height,"depth":depth,"filename":filename
    }
    
def CreateXML(width=0,height=0,depth=3,filename=""):
    xml_node = ET.Element("annotation")
    ET.SubElement(xml_node,"CreateVersion").text = "2.5"
    ET.SubElement(xml_node, "folder").text = ""
    ET.SubElement(xml_node, "filename").text = filename
    ET.SubElement(xml_node, "path").text = ""
    source=ET.SubElement(xml_node, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size=ET.SubElement(xml_node, "size")
    ET.SubElement(size, "width").text = f"{width}"
    ET.SubElement(size, "height").text = f"{height}"
    ET.SubElement(size, "depth").text = f"{depth}"
    ET.SubElement(xml_node, "segmented").text = '0'   
    return xml_node


def Load(xml_content):
    imgdesc = IMGDESC()
    root = ET.fromstring(xml_content)
    header = LoadHeader(root)
    for node in root.findall("object"):
        obj = LoadObject(node)
        if obj.shape_name == "rect":
            x0,y0,x1,y1 = obj.shape_data
            w,h = x1 - x0 + 1, y1 - y0 + 1
            #one pixel as relax
            if w >= header['width'] - 1 and h >= header['height'] and x0 == 0 and y0 == 0:
                imgdesc.add_classes({"class":obj.names[0],'id':obj.ids[0]})
            else:
                imgdesc.objects.append(LoadObject(node))
        else:
            imgdesc.objects.append(LoadObject(node))
    imgdesc.set(header['filename'],header['width'],header['height'],depth=header['depth'])
   
     
    
    return imgdesc

def Save(imgdesc):
    xml_node = CreateXML(imgdesc.width, imgdesc.height, imgdesc.depth, imgdesc.filename)
    for obj in imgdesc.objects:
        if obj.shape_name == "polygon" and len(obj.shape_data) == 4*2:
            obj.shape_name = "quad"
         
        AddObject(xml_node, obj)
    for class_info in imgdesc.classes:
        class_name,class_id = class_info['class'],class_info['id']
        obj = OBJDESC()
        obj.set(class_name, class_id, "rect", (0,0,imgdesc.width-1,imgdesc.height-1),1.0)
        AddObject(xml_node,obj)
    return prettify(xml_node)
    
    
if __name__ == "__main__":
    with open("n:/1.xml","r",encoding='utf-8') as f:
        xml_content = f.read()
    data = Load(xml_content)
    xml_content  = Save(data)
    with open("n:/1a.xml",'w',encoding='utf-8') as f:
        f.write(xml_content) 
        