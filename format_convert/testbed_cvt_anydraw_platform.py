import os,sys
from fmtcvt import fmtcvt
from rich.progress import track

def testbed_anydraw2platform(input_dir):
    xml_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".xml"]
    xml_files = [os.path.join(input_dir,x) for x in xml_files]
    for xml_file in track(xml_files):
        #xml2ai
        with open(xml_file,'r',encoding="utf-8") as f:
            xml_str = f.read()
        ai_data = fmtcvt.convert_from(xml_str,"anydraw")
        with open(os.path.splitext(xml_file)[0] + ".json","w",encoding="utf-8") as f:
            f.write(ai_data)
    return 

def testbed_platform2anydraw(input_dir):
    json_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".json"]
    json_files = [os.path.join(input_dir,x) for x in json_files]
    for json_file in track(json_files):
        #ai2xml
        with open(json_file,"r",encoding="utf-8") as f:
            ai_str = f.read()
        anydraw_data = fmtcvt.convert_to(ai_str,"anydraw")
        with open(os.path.splitext(os.path.splitext(json_file)[0])[0] + ".xml","w",encoding="utf-8") as f:
            f.write(anydraw_data) 
    return 
            
if __name__ == "__main__":
    testbed_platform2anydraw(r'H:\test_data\aiplatform\remote\det')
    #testbed_anydraw2platform(r'H:\test_data\aiplatform\remote\det')