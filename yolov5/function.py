import cv2
import numpy as np
from functools import reduce
import random
import math
BoxInternalType = [[str, str, str, str, str]]  # cls, x, y, w, h
MAX_VALUES_BY_DTYPE = {
                    np.dtype("uint8"): 255,
                    np.dtype("uint16"): 65535,
                    np.dtype("uint32"): 4294967295,
                    np.dtype("float32"): 1.0}


def pipeline(input, funcs):
    return reduce(lambda x, y: y(x), funcs, input)


def is_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def is_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 3


def shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        return img + r_shift
    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = img[..., i] + shift
    return result_img


def shift_image_uint8(img, value):
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    lut = np.arange(0, max_value + 1).astype("float32")
    lut += value
    lut = np.clip(lut, 0, max_value).astype(img.dtype)
    return cv2.LUT(img, lut)


def shift_rgb_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        h, w, c = img.shape
        img = img.reshape([h, w * c])
        return shift_image_uint8(img, r_shift)
    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = shift_image_uint8(img[..., i], shift)
    return result_img


def brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max):
    dtype = np.dtype("float32")
    if alpha != 1:
        img *= alpha
    if beta != 0:
        if beta_by_max:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
            img += beta * max_value
        else:
            img += beta * np.mean(img)
    return img


def brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max):
    dtype = np.dtype("uint8")
    max_value = MAX_VALUES_BY_DTYPE[dtype]
    lut = np.arange(0, max_value + 1).astype("float32")
    if alpha != 1:
        lut *= alpha
    if beta != 0:
        if beta_by_max:
            lut += beta * max_value
        else:
            lut += (alpha * beta) * np.mean(img)
    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


def brightness_contrast_adjust(img, alpha, beta, beta_by_max):
    if img.dtype == np.uint8:
        return brightness_contrast_adjust_uint(img, alpha, beta, beta_by_max)

    return brightness_contrast_adjust_non_uint(img, alpha, beta, beta_by_max)


def channel_shuffle(img):
    ch_arr = list(range(img.shape[2]))
    random.shuffle(ch_arr)
    return img[..., ch_arr]


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def clahe(img, clip_limit, tile_grid_size):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img


def gamma_transform(img, gamma):
    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    else:
        img = np.power(img, gamma)
    return img


def dilate(img, ksize):
    kernel = np.ones(ksize, np.uint8)
    return cv2.dilate(img, kernel)


def erode(img, ksize):
    kernel = np.ones(ksize, np.uint8)
    return cv2.erode(img, kernel)


def blur(img, ksize):

    return cv2.blur(img, tuple(ksize))


def median_blur(img, ksize):
    return cv2.medianBlur(img, ksize)


def motion_blur(img, ksize):
    kernel = np.zeros(ksize, dtype=np.uint8)
    x1, x2 = random.randint(0, ksize[1] - 1), random.randint(0, ksize[1] - 1)
    if x1 == x2:
        y1, y2 = random.sample(range(ksize[0]), 2)
    else:
        y1, y2 = random.randint(0, ksize[0] - 1), random.randint(0, ksize[0] - 1)
    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    kernel = kernel.astype(np.float32) / np.sum(kernel)

    return cv2.filter2D(img, ddepth=-1, kernel=kernel)


def gaussian_blur(img, ksize, sigmaX, sigmaY):
    return cv2.GaussianBlur(img, tuple(ksize), sigmaX=sigmaX, sigmaY=sigmaY)


def sharpen(img, alpha, lightness):
    matrix_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    matrix_effect = np.array([[-1, -1, -1], [-1, 8 + lightness, -1], [-1, -1, -1]], dtype=np.float32)
    matrix = (1 - alpha) * matrix_nochange + alpha * matrix_effect
    return cv2.filter2D(img, ddepth=-1, kernel=matrix)


def gauss_noise(img, mean, sigma, per_channel):
    if per_channel:
        gauss = np.random.normal(mean, sigma, img.shape)
    else:
        gauss = np.random.normal(mean, sigma, img.shape[:2])
        if len(img.shape) == 3:
            gauss = np.expand_dims(gauss, -1)
    
    img = img.astype("float32")
    return (img + gauss).astype(np.float32)


def iso_noise(img, color_shift, intensity):
    one_over_255 = float(1.0 / 255.0)
    img = np.multiply(img, one_over_255, dtype=np.float32)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)
    luminance_noise = np.random.poisson(stddev[1] * intensity * 255, size=hls.shape[:2])
    color_noise = np.random.normal(0, color_shift[0] * 360 * intensity[0], size=hls.shape[:2])
    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360
    luminance = hls[..., 1]
    luminance += (luminance_noise / 255) * (1.0 - luminance)
    img = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * 255
    return img.astype(np.uint8)


def multiply(img, multiplier, per_channel, elementwise):
    if multiplier[0] == multiplier[1]:
        print("multiplier", np.array([multiplier[0]]))
    h, w = img.shape[:2]
    if per_channel:
        c = 1 if (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1) else img.shape[-1]
    else:
        c = 1
    if elementwise:
        shape = [h, w, c]
    else:
        shape = [c]
    multiplier = np.random.uniform(multiplier[0], multiplier[1], shape)
    if is_grayscale_image(img) and img.ndim == 2:
        multiplier = np.squeeze(multiplier)
    if img.dtype == np.uint8:
        if len(multiplier.shape) == 1:
            if is_grayscale_image(img) or len(multiplier) == 1:
                multiplier = multiplier[0]
                lut = np.arange(0, 256, dtype=np.float32)
                lut *= multiplier
                lut = np.clip(lut, 0, MAX_VALUES_BY_DTYPE[img.dtype])
                return cv2.LUT(img, lut.astype(np.uint8))
            channels = img.shape[-1]
            lut = [np.arange(0, 256, dtype=np.float32)] * channels
            lut = np.stack(lut, axis=-1)
            lut *= multiplier
            lut = np.clip(lut, 0, MAX_VALUES_BY_DTYPE[img.dtype])
            images = []
            for i in range(channels):
                img = cv2.LUT(img, lut[:, i].astype(np.uint8))
                images.append(img[:, :, i])
            return np.stack(images, axis=-1)
        return np.multiply(img.astype(np.float32), multiplier).astype(np.float32)
    return img * multiplier


def bbox_flip(img, bboxes, flip_code):
    bboxes_out = []
    height = 1 #img.shape[0]
    width = 1 #img.shape[1]
    for bbox in bboxes:
        _, x, y, w, h = bbox[:5]
        #yolo to voc
        x1 = float(x) - float(w) / 2
        x2 = float(x) + float(w) / 2
        y1 = float(y) - float(h) / 2
        y2 = float(y) + float(h) / 2
        if flip_code == 0:  # 垂直翻转
            y1 = height - float(y1)
            y2 = height - float(y2)
        elif flip_code == 1:  # 水平翻转
            x1 = width - float(x1)
            x2 = width - float(x2)
        elif flip_code == -1:  # 水平垂直翻转
            y1 = height - float(y1)
            y2 = height - float(y2)
            x1 = width - float(x1)
            x2 = width - float(x2)
        X = (x1+x2)/2
        Y = (y1+y2)/2
        bboxes_out.append([_, X, Y, w, h])
    return bboxes_out


def random_affine(img, mask=None, targets=[], degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """
    使用OpenCV对图片进行一列仿射变换：
        随机旋转
        缩放
        平移
        错切
    Args:
        img: 四合一图片 -> img4
        labels：四合一图片的标签 -> labels4
        degrees: 超参数文件中定义的角度（旋转角度） -> 0.0
        translate: 超参数文件中定义的变换方式（平移） -> 0.0
        scale: 超参数文件中定义的scale（缩放） -> 0.0
        shear: 超参数文件中定义的修建（错切） -> 0.0
        border: 这里传入的是（填充大小） -s//2
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，等于img4.shape / 2
    """
        img.shape[0], img.shape[1]为Mosaic相框的宽度和高度（是期待输出图像的两倍）
        因为传入的border=-s//2
            border * 2 -> -s
        所以height和width这个参数和我们期待Mosaic增强的输出是一样的（原图大小而非两倍）
    """
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    """
        @ 表示矩阵相乘（就是传统意义的矩阵相乘而非对应元素相乘）
    """
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        if not mask is None:
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    """
        对图片进行仿射变换后，对它的labels同样也要做对应的变换
    """
    # Transform label coordinates
    n = len(targets)
    T = []
    for i in targets:
        T.append([i[0], (i[1]-i[3]/2)*width, (i[2]-i[4]/2)*height, (i[1]+i[3]/2)*width, (i[2]+i[4]/2)*height])
    targets = np.array(T)
    if n:
        """
            将GTBox4个顶点坐标求出来再进行仿射变换
        """
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)  # 得到经过放射变换后4个顶点的坐标

        """
            求出4个顶点进行仿射变换之后的xy坐标
            取4个顶点的(x_min, y_min)作为新的GTBox的左上角坐标
            取4个顶点的(x_max, y_max)作为新的GTBox的右下角坐标

            为什么这么做呢？
                比如我们的GTBox是一个正常的矩形框，在经过仿射变换后它变成了倾斜的矩形框，但在目标检测中，矩形框一般是正的，不是倾斜的
                所以需要对它的矩形框进行一个重新的调整 -> 这样就求出新的GTBox的合适的坐标了
        """
        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]  # 计算新的GTBox的宽度
        h = xy[:, 3] - xy[:, 1]  # 计算新的GTBox的高度

        # 计算调整后的每个box的面积
        area = w * h

        # 计算调整前的每个box的面积（在对标签仿射变换之前GTBox的面积）
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])

        # 计算仿射变换之后每个GTBox的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio

        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box -> mask
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        # 筛选GTBox
        targets = targets[i]

        # 使用新的GTBox信息替换原来的
        targets[:, 1:5] = xy[i]
    T1 = []
    for i in targets:
        T1.append([i[0], (i[1] + i[3]) / 2 / width, (i[2] + i[4]) / 2 / height, (i[3] - i[1]) / width,
                  (i[4] - i[2]) / height])
    targets = np.array(T1)
    if not mask is None:
        return img, mask
    else:
        return img, targets


def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    img = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
    return img


def bbox_resize(img, bboxes, rh, rw):
    bboxes_out = []
    # voc时要用到
    # height = img.shape[0]
    # width = img.shape[1]
    # for bbox in bboxes:
    #     _, x1, y1, x2, y2 = bbox[:5]
    #     y1 = y1*rh/height
    #     y2 = y2*rh/height
    #     x1 = x1*rw/width
    #     x2 = x2*rw/width
    #     bboxes_out.append([_, x1, y1, x2, y2])
    bboxes_out = bboxes
    return bboxes_out


class RGBShift():
    '''
    色度偏移：
        r_shift_limit: (min,max)
    '''

    def __init__(self, r_shift_limit=[0, 20], g_shift_limit=[], b_shift_limit=[]):
        self.r_shift = random.uniform(r_shift_limit[0], r_shift_limit[1])
        self.g_shift = random.uniform(g_shift_limit[0], g_shift_limit[1])
        self.b_shift = random.uniform(b_shift_limit[0], b_shift_limit[1])

    def apply(self, img):
        if not is_rgb_image(img):
            print("RGBShift transformation expects 3-channel images.")
        if img.dtype == np.uint8:
            return shift_rgb_uint8(img, self.r_shift, self.g_shift, self.b_shift)

        return shift_rgb_non_uint8(img, self.r_shift, self.g_shift, self.b_shift)


class RandomBrightnessContrast():
    '''
    随机调整亮度,对比度：
    '''

    def __init__(self, brightness_limit=[0, 0.2], contrast_limit=[0, 2], brightness_by_max=True):
        self.brightness_by_max = brightness_by_max
        self.alpha = 1.0 + random.uniform(contrast_limit[0], contrast_limit[1])
        self.beta = 0.0 + random.uniform(brightness_limit[0], brightness_limit[1])

    def apply(self, img):
        if img.dtype == np.uint8:
            return brightness_contrast_adjust_uint(img, self.alpha, self.beta, self.brightness_by_max)

        return brightness_contrast_adjust_non_uint(img, self.alpha, self.beta, self.brightness_by_max)


class ChannelShuffle():
    '''
    随机通道洗牌：
    '''

    def apply(self, img):
        return channel_shuffle(img)


class ToGray():
    '''
    转化灰度图
    '''

    def apply(self, img):
        if is_grayscale_image(img):
            print("The image is already gray.")
            return img
        if not is_rgb_image(img):
            print("ToGray transformation expects 3-channel images.")
        return to_gray(img)


class CLAHE():
    '''
    自适应直方图均衡化：
        clip_limit决定均衡的阈值min=1，tile_grid_size决定均衡的图像patch大小
    '''

    def __init__(self, clip_limit=[1, 4], tile_grid_size=[8, 8]):
        self.clip_limit = random.uniform(clip_limit[0], clip_limit[1])
        self.tile_grid_size = tile_grid_size

    def apply(self, img):
        if not is_rgb_image(img) and not is_grayscale_image(img):
            print("CLAHE transformation expects 1-channel or 3-channel images.")

        return clahe(img, self.clip_limit, self.tile_grid_size)


class RandomGamma():
    '''
    随机gmma变换：
        gamma_limit: (min,max)
    '''

    def __init__(self, gamma_limit: list = []):
        self.gamma = random.uniform(gamma_limit[0], gamma_limit[1])

    def apply(self, img):
        return gamma_transform(img, self.gamma)


class Dilation():
    '''
    膨胀：
    '''

    def __init__(self, ksize: list = []):
        self.ksize = ksize

    def apply(self, img):
        return dilate(img, self.ksize)


class Erosion():
    '''
    腐蚀：
    '''

    def __init__(self, ksize: list = []):
        self.ksize = ksize

    def apply(self, img):
        return erode(img, self.ksize)


##############################################Kernel Filters#######################################################

class Blur():
    '''
    模糊变化：
        ksize 滤波核大小
    '''

    def __init__(self, ksize: list = []):
        self.ksize = ksize

    def apply(self, img):
        return blur(img, self.ksize)


class MedianBlur():
    '''
    中值模糊：需要注意，核大小必须是比1大的奇数
        ksize 滤波核大小
    '''

    def __init__(self, ksize: int = 3):
        self.ksize = ksize

    def apply(self, img):
        return median_blur(img, self.ksize)


class MotionBlur():
    '''
    动态模糊：
        ksize 滤波核大小
    '''

    def __init__(self, ksize: list = []):
        self.ksize = ksize

    def apply(self, img):
        return motion_blur(img, self.ksize)


class GaussianBlur():
    '''
    高斯模糊：
        ksize 滤波核大小,
        当不给sigma参数是，会按公式自动计算sigma = 0.3*((ksize-1)*0.5-1)+0.8
    '''

    def __init__(self, ksize: list = [], sigmaX: float = 0.0, sigmaY: float = 0.0):
        self.ksize = ksize
        self.sigmaX, self.sigmaY = sigmaX, sigmaY

    def apply(self, img):
        return gaussian_blur(img, self.ksize, sigmaX=self.sigmaX, sigmaY=self.sigmaY)


class Sharpen():
    '''
    锐化：
        alpha=(min, max), lightness=(min, max)
    '''

    def __init__(self, alpha: float = 1.0, lightness: float = 1.0):
        self.alpha = random.uniform(0, alpha)
        self.lightness = random.uniform(lightness, 1)

    def apply(self, img):
        return sharpen(img, self.alpha, self.lightness)


##############################################Noise#######################################################

class GaussNoise():
    '''
    高斯噪声：
    '''

    def __init__(self, sigma_limit=[10.0, 50.0], mean=0, per_channel=True):
        self.mean = mean
        self.sigma = random.uniform(sigma_limit[0], sigma_limit[1])
        self.per_channel = per_channel

    def apply(self, img):
        return gauss_noise(img, self.mean, self.sigma, self.per_channel)


class ISONoise():
    '''
    传感器噪声：
    '''

    def __init__(self, color_shift=[0.01, 0.05], intensity=[0.01, 0.05]):
        self.color_shift = random.uniform(color_shift[0], color_shift[1]),
        self.intensity = random.uniform(intensity[0], intensity[1]),

    def apply(self, img):
        return iso_noise(img, self.color_shift, self.intensity)


class MultiplicativeNoise():
    '''
    多层偏移噪声：
    '''

    def __init__(self, multiplier=[0.9, 1.1], per_channel=False, elementwise=False):
        self.multiplier = multiplier
        self.per_channel = per_channel
        self.elementwise = elementwise

    def apply(self, img):
        return multiply(img, self.multiplier, self.per_channel, self.elementwise)


class Flip():
    '''
    翻转变化：
        1: 水平翻转
        0: 垂直翻转
        -1:	水平垂直翻转
    '''

    def __init__(self, flip_code: int = 1):
        self.flip_code = flip_code

    def apply(self, img):
        return cv2.flip(img, self.flip_code)

    def apply_to_bbox(self, input):
        img, bbox = input[0], input[1]
        return (cv2.flip(img, self.flip_code), bbox_flip(img, bbox, self.flip_code))

    def apply_to_mask(self, input):
        img, mask = input[0], input[1]
        return (cv2.flip(img, self.flip_code), cv2.flip(mask, self.flip_code))


class Affine():
    """
        使用OpenCV对图片进行一列仿射变换：
            随机旋转
            缩放
            平移
            错切
        Args:
            img: 四合一图片 -> img4
            labels：四合一图片的标签 -> labels4
            degrees: 超参数文件中定义的角度（旋转角度） -> 0.0
            translate: 超参数文件中定义的变换方式（平移） -> 0.0
            scale: 超参数文件中定义的scale（缩放） -> 0.0
            shear: 超参数文件中定义的修建（错切） -> 0.0
            border: 这里传入的是（填充大小） -s//2
        """

    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, border=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border

    def apply(self, img):
        return random_affine(img, degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear,
                               border=self.border)[0]

    def apply_to_bbox(self, input):
        img, bbox = input[0], input[1]
        return (random_affine(img, targets=bbox, degrees=self.degrees, translate=self.translate, scale=self.scale,
                                shear=self.shear,
                                border=self.border))

    def apply_to_mask(self, input):
        img, mask = input[0], input[1]
        return (random_affine(img, mask=mask, degrees=self.degrees, translate=self.translate, scale=self.scale,
                                shear=self.shear,
                                border=self.border))


class Resize():
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img):
        return resize(img, height=self.height, width=self.width, interpolation=self.interpolation)

    def apply_to_bbox(self, input):
        img, bbox = input[0], input[1]
        return (resize(img, height=self.height, width=self.width, interpolation=self.interpolation),
                bbox_resize(img, bbox, rh=self.height, rw=self.width))

    def apply_to_mask(self, input):
        img, mask = input[0], input[1]
        return (resize(img, height=self.height, width=self.width, interpolation=self.interpolation),
                resize(mask, height=self.height, width=self.width, interpolation=self.interpolation))