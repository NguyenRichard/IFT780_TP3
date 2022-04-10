import numpy as np
import math

def identity(x, is_label=False):
    """ Performs the identity transformation on a volume
    """
    return x

def crop(x, is_label=False):

    crop_length = 224
    crop_width = 224
    xheight = x.shape[0]
    xwidth = x.shape[1]
    cwidth = min(crop_length, xwidth)
    cheight= min(crop_length, xheight)
    padheight = xheight-cheight
    padwidth = xwidth-cwidth

    n = np.random.randint(0, x.shape[0]-cwidth)
    m = np.random.randint(0, x.shape[1]-cheight)

    cropped = x[n:n+crop_width, m:m+cheight, :]

    padwbefore = math.floor(padwidth/2)
    padwafter = math.ceil(padwidth/2)
    padhbefore= math.floor(padheight/2)
    padhafter = math.ceil(padheight/2)

    padded = np.pad(cropped, [(padwbefore, padwafter), (padhbefore, padhafter), (0, 0)], constant_values=128)

    return padded

def hflip(x, is_label=False):

    return np.flip(x, axis=0).copy()


def crop_and_hflip(x, is_label=False):
    ''' Returns a random 224x224 cropped version of the image with a horizontal reflection.
        Inspired by the data augmentation of this paper: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf'''
    cropped = crop(x,is_label)
    hflipped = hflip(cropped, is_label)
    return hflipped