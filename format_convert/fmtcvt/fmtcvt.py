from . import json_ai as AiFMT 
from . import pts_dlib as PtsFMT 
from . import xml_anydraw as AnydrawFMT
from . import mask_voc as VocSegFMT
import traceback


class SEGMENT_CONVERT(object):
    def __init__(self,segment_classes = None) -> None:
        self.segfmt = VocSegFMT.SEGMENT_VOC(segment_classes)
        return
    def convert_from(self,mask_data,name2rgb_data, type_name):
        try:
            data_output = ""
            if type_name.lower() == "voc":
                data_input = self.segfmt.Load({"mask":mask_data, "config":name2rgb_data})
            else:
                raise Exception("unk type_name")
            data_output = AiFMT.Save(data_input)
        except Exception as e:
            print(f"!!!ERROR!!! {e}")
            print(f"\t type_name = {type_name}")
            print(f"\t content = {name2rgb_data}")
            traceback.print_exc()
        return data_output
    def convert_to(self,content, type_name):
        try:
            data_output = ""
            data_input = AiFMT.Load(content)
            if type_name.lower() == "voc":
                data_output = self.segfmt.Save(data_input)
            else:
                raise Exception("unk type_name")
        except Exception as e:
            print(f"!!!ERROR!!! {e}")
            print(f"\t type_name = {type_name}")
            print(f"\t content = {content}")
            traceback.print_exc()
        return data_output
    def begin(self):
        return
    def end(self):
        
        return self.segfmt.GetConfig()

def convert_from(content, type_name):
    try:
        data_output = ""
        if type_name.lower() == "pts":
            data_input = PtsFMT.Load(content)
        elif type_name.lower() == "anydraw":
            data_input = AnydrawFMT.Load(content)
        else:
            raise Exception("unk type_name")
        data_output = AiFMT.Save(data_input)
    except Exception as e:
        print(f"!!!ERROR!!! {e}")
        print(f"\t type_name = {type_name}")
        print(f"\t content = {content}")
        traceback.print_exc()
    return data_output


def convert_to(content, type_name):
    try:
        data_output = ""
        data_input = AiFMT.Load(content)
        if type_name.lower() == "pts":
            data_output = PtsFMT.Save(data_input)
        elif type_name.lower() == "anydraw":
            data_output = AnydrawFMT.Save(data_input)
        else:
            raise Exception("unk type_name")
    except Exception as e:
        print(f"!!!ERROR!!! {e}")
        print(f"\t type_name = {type_name}")
        print(f"\t content = {content}")
        traceback.print_exc()
    return data_output

   


