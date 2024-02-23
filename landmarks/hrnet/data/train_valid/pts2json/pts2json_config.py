from yacs.config import CfgNode as CN

_C = CN()
_C.INFO = CN()
_C.INFO.TOP_RANTGE = 799
_C.INFO.BOTTOM_RANGE = 300
_C.INFO.LEFT_RANGE = 39
_C.INFO.RIGHT_RANGE = 40
_C.INFO.RATION = 2.65                               #width/height
_C.INFO.NUM_POINTS = 30
_C.INFO.SRC_ROOT = "/home1/landmarks/HRNet/hrnetw18_landmarks/data/mpii/imagesB_38"
_C.INFO.DST_ROOT = "/home1/landmarks/HRNet/hrnetw18_landmarks/data/mpii/annot"
_C.INFO.JSON_NAME = "train.json"
_C.INFO.VIS_FLAG = True

def get_default_config():
    return _C.clone()
    
    
    



##################   顶部全局相机    ##########
###------------------外圈————————————————————
#INFO:
#    TOP_RANTGE: 150
#    BOTTOM_RANGE: 200
#    LEFT_RANGE: 300
#    RIGHT_RANGE: 300
#    RATION: 1.0
#    NUM_POINTS: 60



##################  顶部边缘油污相机   ##########
###———————————————————丝面边缘——————————————————
#    TOP_RANTGE: 799
#    BOTTOM_RANGE: 300
#    LEFT_RANGE: 39
#    RIGHT_RANGE: 40
#    RATION: 2.65
#    NUM_POINTS: 30
