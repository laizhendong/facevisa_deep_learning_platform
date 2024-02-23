import os,sys
from fmtcvt import fmtcvt
from rich.progress import track


def testbed_pts2platform(input_dir):
    pts_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".pts"]
    pts_files = [os.path.join(input_dir,x) for x in pts_files]
    for pts_file in track(pts_files):
        #pts2ai
        with open(pts_file,'r',encoding="utf-8") as f:
            pts_str = f.read()
        ai_data = fmtcvt.convert_from(pts_str,"pts")
        with open(os.path.splitext(pts_file)[0] + ".json","w",encoding="utf-8") as f:
            f.write(ai_data)
    return 

def testbed_platform2pts(input_dir):
    json_files = [x for x in os.listdir(input_dir) if os.path.splitext(x)[-1].lower() == ".json"]
    json_files = [os.path.join(input_dir,x) for x in json_files]
    for json_file in track(json_files):
        #ai2pts
        with open(json_file,"r",encoding="utf-8") as f:
            json_str = f.read()
        pts_data = fmtcvt.convert_to(json_str,"pts")
        with open(os.path.splitext(os.path.splitext(json_file)[0])[0]+".pts","w",encoding="utf-8") as f:
            f.write(pts_data)
    return 
    
if __name__ == "__main__":
    testbed_platform2pts(r'H:\test_data\aiplatform\remote\pts')
    testbed_pts2platform(r'H:\test_data\aiplatform\remote\pts')