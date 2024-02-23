import json


class MODELCARD:
    version = 1
    task = ""
    model_name = ""
    model_id = 0,
    input_shape = []
    classes_local2global = {}
    mean = []
    std = []
    input_blobs = []
    output_blobs = []
    privates = {}
    def __init__(self,task,version=1) -> None:
        self.version = version
        self.task = task
        return
    def set_mean_std(self,mean,std):
        self.mean = mean 
        self.std = std
        return 
    def set_classes(self,local2global, global2name):
        self.classes_local2global = {}
        for local in local2global.keys():
            glb = global2name[local2global[local]]
            self.classes_local2global["{}".format(local)] = "{}".format(glb)
        self.privates["global2name"] = global2name
        return
            
    def set_inputs(self,names, shapes):
        self.input_shape, self.input_blobs = [], []
        if len(names) > 1:
            for name,shape in zip(names, shapes):
                self.input_blobs.append(name)
                self.input_shape.append(shape)
        else:
            self.input_blobs = names[0]
            self.input_shape = shapes[0]
        return
    def set_outputs(self,names):
        self.output_blobs = []
        if len(names) > 1:
            for name in names:
                self.output_blobs.append(name)
        else:
            self.output_blobs = names[0]
        return
    def set_name(self,name):
        self.model_name = name.lower()
        if self.model_name == "resnet18":
            self.model_id = 2
        elif self.model_name == 'unet_vgg16':
            self.model_id = 3
        else:
            self.model_id = 0
        return
    def get_class_name(self,global_class_id):
        if "global2name" in self.privates.keys():
            return self.privates["global2name"][global_class_id]
        return "NONE"
    def convert_name2global(self):
        if 'global2name' not in self.privates: #compatible to older version 
            return 
        name2global = {}
        for glb in self.privates['global2name'].keys():
            name = self.privates['global2name'][glb]
            name2global[name] = glb
        results = {}
        for local in self.classes_local2global.keys():
            name = self.classes_local2global[local]
            results[local] = name2global[name]
        self.classes_local2global = results
        return
    def tojson(self):
        card = {
            "version": self.version,
            "task": self.task,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "input_shape": self.input_shape,
            "classes": self.classes_local2global,
            "mean": self.mean,
            "std": self.std,
            "input_blobs": self.input_blobs,
            "output_blobs": self.output_blobs,
            "privates" : self.privates
        }
        return json.dumps(card,ensure_ascii=False,indent=4)
    def fromjson(self,json_data_raw):
        json_data = json.loads(json_data_raw)
        self.version = json_data['version']
        self.task = json_data['task']
        self.model_name = json_data['model_name']
        self.model_id = json_data['model_id']
        self.input_shape = json_data['input_shape']
        self.classes_local2global = json_data['classes']
        self.input_blobs = json_data['input_blobs']
        self.output_blobs = json_data['output_blobs']
        self.mean = json_data['mean']
        self.std = json_data['std']
        if "privates" in json_data.keys():
            self.privates = json_data["privates"]
        else:
            print("warning: load card in older format")
            self.privates = {}
        return