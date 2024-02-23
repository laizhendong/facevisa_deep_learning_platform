import os,sys
from fmtcvt.fmtcvt import SEGMENT_CONVERT
from rich.progress import track


segment_config_file = "segment_colors.json"

def testbed_segment2platform(input_dir):
    try:
        with open(os.path.join(input_dir,segment_config_file),"r",encoding="utf-8") as f:
            config_data = f.read()
    except Exception as e:
        print(e)
             
    png_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".png"]
    png_files = [os.path.join(input_dir,x) for x in png_files]
    
    convertor = SEGMENT_CONVERT()
    convertor.begin() #####################
    for png_file in track(png_files):
        #png2ai
        with open(png_file,'rb') as f:
            png_data = f.read()
        ai_data = convertor.convert_from(png_data,config_data,"voc")
        with open(os.path.splitext(png_file)[0] + ".json","w",encoding="utf-8") as f:
            f.write(ai_data)
    convertor.end() #####################
    return 

#debug only
def debug_scan_for_classes(input_dir):
    import json
    json_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".json" and x != segment_config_file]
    json_files = [os.path.join(input_dir,x) for x in json_files]   
    classnames = []
    for json_file in json_files:
        with open(json_file,'r',encoding='utf-8') as f:
            content = json.load(f)
        try:
            for inst in content['instances']:
                for att in inst['attributes']:
                    classnames.append(att['class_name'])
        except Exception as e:
            print(e)
            print('skip ',os.path.basename(json_file))
    return list(set(classnames))
        

def testbed_platform2segment(input_dir):
    classnames = debug_scan_for_classes(input_dir)
    convertor = SEGMENT_CONVERT(classnames)
    convertor.begin() ###################################################
    json_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".json" and x != segment_config_file]
    json_files = [os.path.join(input_dir,x) for x in json_files]
    for json_file in track(json_files):
        #ai2xml
        with open(json_file,"r",encoding="utf-8") as f:
            ai_str = f.read()
        png_data = convertor.convert_to(ai_str,"voc")
        with open(os.path.splitext(os.path.splitext(json_file)[0])[0] + ".png","wb") as f:
            f.write(png_data) 
    config_data = convertor.end() ####################################################
    with open(os.path.join(input_dir,segment_config_file),"w",encoding='utf-8') as f:
        f.write(config_data)
    return 
            
if __name__ == "__main__":
    testbed_segment2platform(r'H:\test_data\aiplatform\remote\seg\voc')
    testbed_platform2segment(r'H:\test_data\aiplatform\remote\seg\voc')