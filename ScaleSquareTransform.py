import torch
from PIL import Image, ImageOps
import collections

class ScaleSquare(object):
    """Rescales the input PIL.Image to the given 'size'.
    size: size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.LANCZOS):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)