from yacs.config import CfgNode as CN

_C = CN()

##############################################
#Model
_C.MODEL = CN()
_C.MODEL.BACKBONE_NAME = ""
_C.MODEL.WEIGHTS = ""
_C.MODEL.USE_IMAGENET_BACKBONE = True

############################################
#Dataset
_C.DATA = CN()
_C.DATA.SRC_DIR = ""
_C.DATA.NAME = ""
_C.DATA.CLASSES_BALANCED = False
_C.DATA.MEAN = (0,0,0)
_C.DATA.STD = (1,1,1)
_C.DATA.THREAD_NUM = 8
_C.DATA.PREFETCH_FACTOR = 32
_C.DATA.TRAIN_GRAY_IMAGE = False
#Augment
_C.DATA.AUGMENT = CN()
_C.DATA.AUGMENT.SHEAR = (0,0)
_C.DATA.AUGMENT.BLUR = -1
_C.DATA.AUGMENT.GAMMA = 0.0
_C.DATA.AUGMENT.ROTATION = -1.0
_C.DATA.AUGMENT.HFLIP = False
_C.DATA.AUGMENT.VFLIP = False
_C.DATA.AUGMENT.COLOR = False
_C.DATA.AUGMENT.SHUFFLE_COLOR = False
_C.DATA.AUGMENT.RESIZE = (0,0) #W,H
_C.DATA.AUGMENT.CROP = (0,0) #W,H
_C.DATA.AUGMENT.RANDOMBRIGHTNESS = [-1.0,-1.0,"add"]
_C.DATA.AUGMENT.RANDOMCONTRAST = [-1.0, -1.0]
_C.DATA.AUGMENT.JPEGCOMPRESSION = [-1,-1]
_C.DATA.AUGMENT.RESIZE_METHOD = ["linear","area"]
######################################
#SOLVER
_C.SOLVER = CN()
_C.SOLVER.LR_BASE = 0.001
_C.SOLVER.LR_POLICY = "cosine"
_C.SOLVER.GRADIENT_MAX = -1
_C.SOLVER.EPOCHS = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.DEVICE = 1


def get_default_config():
    return _C.clone()


