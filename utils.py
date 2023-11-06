import numpy as np 
import os
import torch 
from torchvision import datasets
from torchvision.transforms import v2
from PIL import Image

def take_file_name(filedir:str) -> str: 
    """
    return just the file name from a directory
    """
    # filename = np.array(filedir.split('/'))[-1].split('.')[0] # take out the name, isolate the jpeg, then return the name
    filename = np.array(filedir.split('/'))[-1] # take out the name, then return the name
    # print(filename)
    return filename

def match_file_names(watermarkedarr:np.ndarray, nonwatermarkedarr:np.ndarray, dname_wm:str, dname_nwm:str) -> list:
    """
    take two watermark/nowatermark lists respectively, sort them so that filenames in each match
    """
    sortedwmarr = np.array([])
    sortednwmarr = np.array([])
    
    wmarr = list(watermarkedarr)
    nwmarr = list(nonwatermarkedarr)
    
    length = len(watermarkedarr) if len(watermarkedarr) >= len(nonwatermarkedarr) else len(nonwatermarkedarr)
    
    for pos in range(length):
        try:
            if length == len(watermarkedarr): # more images in watermarked array
                exist_nwm = nwmarr.index(wmarr[pos])
                sortedwmarr = np.append(sortedwmarr, dname_wm + watermarkedarr[pos]) # this is the iterable
                sortednwmarr = np.append(sortednwmarr, dname_nwm + nonwatermarkedarr[exist_nwm]) # this is the match
            elif length == len(nonwatermarkedarr): # more images in nonwatermarked array
                exist_wm = wmarr.index(nwmarr[pos])
                sortedwmarr = np.append(sortedwmarr, dname_wm + watermarkedarr[exist_wm]) # this is the match
                sortednwmarr = np.append(sortednwmarr, dname_nwm + nonwatermarkedarr[pos]) # this is the iterable
        except ValueError: 
            continue
    return sortedwmarr, sortednwmarr

def get_paths(split:str='train') -> list:
    """
    returns unsorted file paths of watermarked/nonwatermarked images for a given split (train, val to be implemented) 
    """
    train_wmark_path = 'C:\\Users\\death\\Desktop\\rnns\\train\\no-watermark\\'
    train_nowmark_path = 'C:\\Users\\death\\Desktop\\rnns\\train\\watermark\\'

    train_wmark = np.array([])
    train_nowmark = np.array([])

    for root, dirs, files in os.walk(train_wmark_path, topdown=True): # data length = 12510
        for f in files:
            train_wmark = np.append(train_wmark, take_file_name(f)) # append just the name of the file into the array
    
    for root, dirs, files in os.walk(train_nowmark_path, topdown=True): # data length = 12477
        for f in files:
            train_nowmark = np.append(train_nowmark, take_file_name(f)) # append just the name of the file into the array

    return train_wmark, train_nowmark, train_wmark_path, train_nowmark_path


def process_dataset(dataset:list) -> list:
    """
    returns a tensor of all the images in the dataset, converted to tensors and randomly augmented\n
    dataset -- a list of file paths 
    """
    out = []

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(128),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

    for path in dataset:
        im = Image.open(path)
        im_tensor = transform(im)
        out.append(im_tensor)
    
    return out 
