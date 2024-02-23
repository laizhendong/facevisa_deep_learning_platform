import os,sys
import copy

SHAPE_TYPES = ["rect","polygon","points","pts"]

class OBJDESC(object):
    def __init__(self) -> None:
        self.names = []
        self.ids:list[int] = []
        self.scores:list[float] = []
        self.shape_data:list[int] = []
        self.shape_name = ""
        return
    def set(self,name,name_id,shape_name, shape_data, confidence = 1.0):
        assert(isinstance(shape_data,(list,tuple))), "shape data must be list or tuple"
        self.names.append(name)
        self.ids.append(int(name_id)) 
        self.scores.append(float( confidence ))
        
        self.shape_name = shape_name.lower()
        self.shape_data = copy.deepcopy(shape_data)
        return
    
class IMGDESC(object):
    def __init__(self) -> None:
        self.objects = []
        self.classes = [] #classification
        self.filename = ""
        self.width:int  = 0
        self.height:int = 0
        self.depth:int = 0
        return
    def set(self,filename,width,height,depth):
        self.filename = filename
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        return
    def add_object(self,object):
        self.objects.append(object)
        return
    def add_classes(self,classes_info):
        if isinstance(classes_info,(list,tuple)):
            for info in classes_info:
                self.classes.append(
                    {"class":info['class'],"id":info['id']}
                )
        elif isinstance(classes_info,(dict,)):
            self.classes.append(
                {"class":classes_info['class'], "id":classes_info['id']}
            )
        return
    