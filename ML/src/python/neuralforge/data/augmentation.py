import torch
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from typing import List, Tuple

class RandAugment:
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.invert, 0, 1),
            (self.rotate, 0, 30),
            (self.posterize, 0, 4),
            (self.solarize, 0, 256),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, 0, 0.3),
            (self.shear_y, 0, 0.3),
            (self.translate_x, 0, 0.3),
            (self.translate_y, 0, 0.3),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img
    
    @staticmethod
    def auto_contrast(img, _):
        return ImageOps.autocontrast(img)
    
    @staticmethod
    def equalize(img, _):
        return ImageOps.equalize(img)
    
    @staticmethod
    def invert(img, _):
        return ImageOps.invert(img)
    
    @staticmethod
    def rotate(img, magnitude):
        return img.rotate(magnitude)
    
    @staticmethod
    def posterize(img, magnitude):
        magnitude = int(magnitude)
        return ImageOps.posterize(img, magnitude)
    
    @staticmethod
    def solarize(img, magnitude):
        return ImageOps.solarize(img, int(magnitude))
    
    @staticmethod
    def color(img, magnitude):
        return ImageEnhance.Color(img).enhance(magnitude)
    
    @staticmethod
    def contrast(img, magnitude):
        return ImageEnhance.Contrast(img).enhance(magnitude)
    
    @staticmethod
    def brightness(img, magnitude):
        return ImageEnhance.Brightness(img).enhance(magnitude)
    
    @staticmethod
    def sharpness(img, magnitude):
        return ImageEnhance.Sharpness(img).enhance(magnitude)
    
    @staticmethod
    def shear_x(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    
    @staticmethod
    def shear_y(img, magnitude):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
    
    @staticmethod
    def translate_x(img, magnitude):
        magnitude = magnitude * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
    
    @staticmethod
    def translate_y(img, magnitude):
        magnitude = magnitude * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))

class MixUp:
    def __init__(self, alpha: float = 1.0, num_classes: int = 1000):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class CutMix:
    def __init__(self, alpha: float = 1.0, num_classes: int = 1000):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(self, images, labels):
        batch_size = images.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(images.device)
        
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, labels, labels[index], lam

class GridMask:
    def __init__(self, d1: int = 96, d2: int = 224, rotate: float = 1, ratio: float = 0.5):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
    
    def __call__(self, img):
        h, w = img.shape[-2:]
        
        d = np.random.randint(self.d1, self.d2)
        l = int(d * self.ratio + 0.5)
        
        mask = np.ones((h, w), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        for i in range(h // d + 1):
            s_h = d * i + st_h
            t_h = min(s_h + l, h)
            for j in range(w // d + 1):
                s_w = d * j + st_w
                t_w = min(s_w + l, w)
                mask[s_h:t_h, s_w:t_w] = 0
        
        mask = torch.from_numpy(mask).to(img.device)
        img = img * mask
        
        return img

class RandomErasing:
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4, r1: float = 0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                
                img[0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                img[1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                img[2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                
                return img
        
        return img
