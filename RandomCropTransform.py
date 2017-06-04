import torch
from numpy.random import randint
from PIL import Image, ImageOps
import collections

class RandomCrop(object):
    """Rescales the input PIL.Image to the given 'size'.
    size: size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        wfree = img.size[0] - self.size[0]
        hfree = img.size[1] - self.size[1]
        rwoffset = randint(0,wfree+1)
        rhoffset = randint(0,hfree+1)
        return img.crop((wfree,hfree,self.size[0]+wfree,self.size[1]+hfree))