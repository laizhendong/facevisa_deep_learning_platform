
import cv2,os
import numpy as np
import random
import math



class RandomBlur(object):
    def __init__(self,cfg):
        super(RandomBlur, self).__init__()
        self._win_size = 0
        self._flag_on = False
        self._win_size = cfg.DATA.AUGMENT.BLUR
        if self._win_size > 2:
            self._flag_on = True


    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        img = x[0].astype(np.uint8)

        blur_type = random.randint(0,3)
        blur_param = 1 + 2 * random.randint(1,self._win_size//2)

        if blur_type == 0:
            img = cv2.GaussianBlur(img,(blur_param,blur_param),0)
        elif blur_type == 1:
            img = cv2.blur(img,(blur_param,blur_param))
        elif blur_type == 2:
            img = cv2.medianBlur(img,blur_param)
        elif blur_type == 3:
            img = cv2.boxFilter(img,-1,(blur_param,blur_param))

        return img, x[1]




class Resize(object):
    def __init__(self,cfg, random_method):
        super(Resize, self).__init__()
        self._width = 0
        self._height = 0
        self._inter = 1
        self._random_method = random_method
        self._name = "resize"
        self._flag_on = False
        self._width, self._height = cfg.DATA.AUGMENT.RESIZE
        self._resize_methods = []
        for m in cfg.DATA.AUGMENT.RESIZE_METHOD:
            lm = m.lower()
            if lm == "linear":
                self._resize_methods.append(cv2.INTER_LINEAR)
            elif lm == "area":
                self._resize_methods.append(cv2.INTER_AREA)
            elif lm == "nn":
                self._resize_methods.append(cv2.INTER_NN)
            else:
                assert(f"unk {lm} for resize")
                
        if self._width > 0 and self._height > 0:
            self._flag_on = True


    def forward(self, x):
        if not self._flag_on:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)
        imgh,imgw = img.shape[0:2]
        if imgh != self._height or imgw != self._width:
            if self._random_method and self._resize_methods != []:
                interpolation = random.choice(self._resize_methods)
            else:
                interpolation = cv2.INTER_LINEAR  #follow trt inference lib  
            img = cv2.resize(img,(self._width, self._height),interpolation=interpolation)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]
        return img



class Crop(object):
    def __init__(self,cfg,random_crop=False):
        super(Crop, self).__init__()
        self._width = 0
        self._height = 0
        self._flag_on = False
        self._random_crop = random_crop
        self._width, self._height = cfg.DATA.AUGMENT.CROP
        if self._width > 0 and self._height > 0:
            self._flag_on = True


    def forward(self, x):
        if not self._flag_on:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0]
        else:
            img = x

        H,W = img.shape[0], img.shape[1]
        if self._random_crop:
            dy,dx = random.randint(0,(H-self._height)//2), random.randint(0,(W-self._width)//2)
        else:
            dy,dx = (H-self._height)//2, (W-self._width)//2

        img = img[dy:dy+self._height,dx:dx+self._width]

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]
        return img

class RandomShear(object):
    def __init__(self,cfg):
        super(RandomShear, self).__init__()
        self._dx,self._dy = 0,0
        self._flag_on = False

        self._dx, self._dy = cfg.DATA.AUGMENT.SHEAR
        if self._dx > 0 or self._dy > 0:
            self._flag_on = True

    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        img = x[0].astype(np.uint8)
        dx = random.uniform(-1 * self._dx, self._dx)
        dy = random.uniform(-1 * self._dy, self._dy)
        M = np.zeros((2,3))
        M[0,0] = 1
        M[1,1] = 1
        M[0,1] = dx
        M[1,0] = dy
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT)

        return img, x[1]



class RandomRotation(object):
    def __init__(self,cfg):
        super(RandomRotation, self).__init__()
        self._degree = 0
        self._flag_on = False

        self._degree = cfg.DATA.AUGMENT.ROTATION
        if self._degree > 0:
            self._flag_on = True
            print("WARNING: ROTATION is slow for large image!!!!!!!!!!")

    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)

        H,W = img.shape[0],img.shape[1]
        #cv2.circle(img, (W//2,H//2),5,(255,0,0),3)
        degree = random.uniform(-1.0 * self._degree, self._degree * 1.0)
        M = cv2.getRotationMatrix2D((W/2,H/2),degree,1.0)

        points = np.asarray(
            [[0,0,1],[W-1,0,1],[W-1,H-1,1],[0,H-1,1]]
        ).transpose()
        points_rot = np.matmul(M,points)
        bbox = [ np.min(points_rot[0,:]), np.min(points_rot[1,:]), np.max(points_rot[0,:]), np.max(points_rot[1:]) ]
        bbox[2] -= bbox[0]
        bbox[3] -= bbox[1]
        bbox = [int(x) for x in bbox]
        M[0,2] += bbox[2] / 2 - W/2
        M[1,2] += bbox[3] / 2 - H/2
        img_new = cv2.warpAffine(img, M, (W,H), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)

        #cv2.imshow("new",img_new)
        #cv2.imshow("src",img)
        #cv2.waitKey(-1)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img_new, x[1]
        return img_new


class RandomLight(object):
    def __init__(self,cfg):
        super(RandomLight, self).__init__()
        self._gamma_delta = -1
        self._flag_on = False
        self.luts = []
        self._gamma_delta = cfg.DATA.AUGMENT.GAMMA
        if self._gamma_delta > 0:
            self._flag_on = True
            lower = 1 - self._gamma_delta
            upper = 1 + self._gamma_delta
            bins = 10
            for step in range(bins+1):
                gamma = lower + (upper - lower) * step / bins
                lut = []
                for l in range(256):
                    lut.append(int(math.pow(l / 255.0,gamma) * 255))
                lut = np.clip(lut,0,255).astype(np.uint8)
                self.luts.append(lut)


    def forward(self, x):
        if not self._flag_on or random.uniform(0,1) < 0.5:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)   
                 
        index = random.randint(0, len(self.luts)-1)
        lut = self.luts[index]
        img = cv2.LUT(img,lut)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]
        return img


class RandomHFlip(object):
    def __init__(self,cfg):
        super(RandomHFlip, self).__init__()
        self._name = "hflip"
        self._flag_on = cfg.DATA.AUGMENT.HFLIP

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)           
        #img = x[0]
        img = cv2.flip(img,1)
        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]        
        return img

class RandomVFlip(object):
    def __init__(self,cfg):
        super(RandomVFlip, self).__init__()

        self._flag_on = cfg.DATA.AUGMENT.VFLIP

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x
        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)   
        img = cv2.flip(img,0)
        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]
        return img


class RandomColor(object):
    def __init__(self,cfg):
        super(RandomColor, self).__init__()

        self._flag_on = cfg.DATA.AUGMENT.COLOR

    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x

        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0].astype(np.uint8)
        else:
            img = x.astype(np.uint8)   
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0].astype(np.int32)
        hue =  ((hue + random.randint(0,360)) % 180).astype(np.uint8)
        hsv[:, :, 0] = hue
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img, x[1]        
        return img

class ShuffleColor(object):
    def __init__(self,cfg):
        super(ShuffleColor, self).__init__()

        self._flag_on = cfg.DATA.AUGMENT.SHUFFLE_COLOR
    def forward(self, x):
        if not self._flag_on or random.uniform(0, 1) < 0.5:
            return x

        if isinstance(x,(list,tuple)) and len(x) == 2:
            img = x[0]
        else:
            img = x
        perm = [img[:,:,0], img[:,:,1], img[:,:,2]]
        random.shuffle(perm)

        img_new = np.stack(perm,axis=2)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return img_new, x[1]
        return img_new

class RandomContrast(object):
    def __init__(self, cfg):
        super(RandomContrast, self).__init__()
        self.lower = cfg.DATA.AUGMENT.RANDOMCONTRAST[0]
        self.upper = cfg.DATA.AUGMENT.RANDOMCONTRAST[1]
        self.flag_on = False
        if self.upper > self.lower and self.lower >= 0:
            self.flag_on = True

    # expects float image
    def forward(self, x):
        if not self.flag_on or random.uniform(0,1) < 0.5:
            return x
        
        if isinstance(x,(list,tuple)) and len(x) == 2:
            image = x[0].astype(np.uint8)
        else:
            image = x.astype(np.uint8)

        alpha = random.uniform(self.lower, self.upper)
        image = image.astype(np.float32)
        for c in range(image.shape[-1]):
            image[:,:,c] = image[:,:,c] + (image[:,:,c] - np.mean(image[:,:,c])) * alpha
        image = np.clip(image,0,255).astype(np.uint8)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return image, x[1]
        return image


class RandomBrightness(object):
    def __init__(self, cfg,):
        self.lower   = cfg.DATA.AUGMENT.RANDOMBRIGHTNESS[0]
        self.upper = cfg.DATA.AUGMENT.RANDOMBRIGHTNESS[1]
        self.type = cfg.DATA.AUGMENT.RANDOMBRIGHTNESS[2].lower()
        self.flag_on = False
        if self.upper > self.lower:
            self.flag_on = True


    def forward(self, x):
       
        if not self.flag_on: 
            return x
        
        if isinstance(x,(list,tuple)) and len(x) == 2:
            image = x[0].astype(np.uint8)
        else:
            image = x.astype(np.uint8)

        if random.randint(0,1):
            if self.type == 'add':
                delta = random.uniform(self.lower, self.upper)
                image = image.astype(np.float32)
                image += delta
                image = np.clip(image,0,255).astype(np.uint8)
            elif self.type == 'multiple':
                ratio = random.uniform(self.lower, self.upper)
                image = image.astype(np.float32)
                image *= ratio
                image = np.clip(image,0,255).astype(np.uint8)
            else:
                assert(0),"unk type in RandomBrightness: {}".format(self.type)

        if isinstance(x,(list,tuple)) and len(x) == 2:
            return image, x[1]
        return image
    
class JPEGCompression(object):
    def __init__(self, cfg):
        super(JPEGCompression, self).__init__()
        self.low,self.high = cfg.DATA.AUGMENT.JPEGCOMPRESSION
        self.low, self.high = int(100 * self.low), int(self.high * 100)
        self.flag_on = False
        if self.low >= 0 and self.high > self.low:
            self.flag_on = True
            assert self.low >= 0 and self.low <= 100
            assert self.high >= self.low and self.high <= 100

    def forward(self, x):
        if not self.flag_on or random.uniform(0, 1) < 0.5:
            return x
        
        if isinstance(x,(list,tuple)) and len(x) == 2:
            image = x[0].astype(np.uint8)
        else:
            image = x.astype(np.uint8)

        quality = int(random.randint(self.low,self.high))
        _, image = cv2.imencode(".jpeg",image,(cv2.IMWRITE_JPEG_QUALITY,quality))
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        if isinstance(x,(list,tuple)) and len(x) == 2:
            return image, x[1]
        return image
    
class ToTensor(object):
    def __init__(self, mean,std):
        super(ToTensor, self).__init__()
        self.mean_ = np.asarray(mean).reshape((1,1,-1))
        self.std_  = np.asarray(std).reshape((1,1,-1))
        return

    def forward(self, x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            img = (x[0]  - self.mean_) / self.std_
            return img,x[1]
        else:
            img = (x - self.mean_) / self.std_
            return img


if __name__=="__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),".."))
    from config.default_config import get_default_config

    img = cv2.imread("n:/5a.jpg",1)
    cfg = get_default_config()
    #cfg.DATA.AUGMENT.JPEGCOMPRESSION = [0.5,0.55]
    #trans = JPEGCompression(cfg)
    #cfg.DATA.AUGMENT.RANDOMBRIGHTNESS = [0.2,0.5,'multiple']
    #cfg.DATA.AUGMENT.RANDOMBRIGHTNESS = [-10,100,'add']
    #trans = RandomBrightness(cfg)
    cfg.DATA.AUGMENT.RANDOMCONTRAST = [0.5,1.5]
    trans = RandomContrast(cfg)
    #cfg.DATA.AUGMENT.COLOR = True
    #trans = RandomColor(cfg)
    for k in range(6):
        en = trans.forward(img)
        cv2.imshow("en",en)
        cv2.imwrite(f"n:/{k}.png",en)
        cv2.waitKey(-1)
    print("done")
