#Copyright (c) 2020 Pierre David

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
from flow_io import flow_read
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
from flow_interpolation import flow_interp

def _make_dataset(dir):
    """
    Creates a 2D list of all the frames in N clips.

    2D List Structure:
    [[im0, imt, im1, f01, f10, ft1]  <-- clip0
     [im0, imt, im1, f01, f10, ft1]  <-- clip0
     :
     [im0, imt, im1, f01, f10, ft1]] <-- clipN

    Parameters
    ----------
        dir : string
            root directory containing clips.

    Returns
    -------
        list
            2D list described above.
    """


    framesPath = []
    # Find and loop over all the clips in root `dir`.
    folders = [('im0', '.png'), ('imt', '.png'), ('im1', '.png'), ('f01', '.flo'), ('f10', '.flo'), ('ft1', '.flo')]
    
    for index, frame in enumerate(sorted(os.listdir(os.path.join(dir, 'im0')))):
        framesPath.append([])
        root, file_extension = os.path.splitext(frame)
        for folder, ext in folders:
            filename = root + ext
            framesPath[index].append(os.path.join(dir, folder, filename))
    return framesPath

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens image at `path` using pil and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            image
    """


    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else img
        # Resize image if specified.
        resized_img = cropped_img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else cropped_img
        # Flip image horizontally if specified.
        flipped_img = resized_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else resized_img
        return flipped_img.convert('RGB')
    
def _flo_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Opens flow at `path` using flow_io and applies data augmentation.

    Parameters
    ----------
        path : string
            path of the image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            flow
    """

    u,v = flow_read(path)
    # Flip image horizontally if specified.
    if frameFlip:
        u = -cv.flip(u, 1)
        v =  cv.flip(v, 1)

    # Crop image if crop area specified.
    if cropArea != None:
        u = u[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]].copy()
        v = v[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]].copy()
    
    # Resize image if specified.
    if resizeDim != None:
        origDim = u.shape
        scale_u = resizeDim[0]/origDim[1]
        scale_v = resizeDim[1]/origDim[0]
        u = scale_u*cv.resize(u, dsize=(resizeDim[0], resizeDim[1]), interpolation=cv.INTER_LANCZOS4)
        v = scale_v*cv.resize(v, dsize=(resizeDim[0], resizeDim[1]), interpolation=cv.INTER_LANCZOS4)
 
    flow = np.stack((u,v), axis=-1)
    return flow

class LFSloMo(data.Dataset):
    """
    A dataloader for loading N samples arranged in this way:

        |-- im0
            |-- frame00
            |-- frame01
            :
            |-- frameN
        |-- imt
            |-- frame00
            |-- frame01
            :
            |-- frameN
        |-- im1
            |-- frame00
            |-- frame01
            :
            |-- frameN
        |-- f01
            |-- frame00
            |-- frame01
            :
            |-- frameN
        |-- f10
            |-- frame00
            |-- frame01
            :
            |-- frameN
        |-- ft1
            |-- frame00
            |-- frame01
            :
            |-- frameN

    ...

    Attributes
    ----------
    framesPath : list
        List of frames' path in the dataset.

    Methods
    -------
    __getitem__(index)
        Returns the sample corresponding to `index` from dataset.
    __len__()
        Returns the size of dataset. Invoked as len(datasetObj).
    __repr__()
        Returns printable representation of the dataset object.
    """


    def __init__(self, root, transform=None, dim=(1024, 436), randomCropSize=(768, 416), train=0):
        """
        Parameters
        ----------
            root : string
                Root directory path.
            transform : callable, optional
                A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            dim : tuple, optional
                Dimensions of images in dataset. Default: (640, 360)
            randomCropSize : tuple, optional
                Dimensions of random crop to be applied. Default: (352, 352)
            train : int, optional
                Specifies if the dataset is for training (0), validation (1) or testing (2)
                `0` returns samples with data augmentation like random 
                flipping, random cropping, etc. while `1` returns the
                samples without randomization but with ground truth, 
                '2' is like '1' except the ground truth is empty. Default: 0
        """


        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.framesPath     = framesPath
        
        if train % 3 == 0:
            self.origDim = randomCropSize
            self.dim = randomCropSize
        else:      
            frame        = _pil_loader(framesPath[0][0])
            self.origDim = frame.size
            self.dim     = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32

        

    def __getitem__(self, index):
        """
        Returns the sample corresponding to `index` from dataset.

        The sample consists of two reference frames - I0 and I1 -
        an intermediate frame - It - forward and backward optical
        flows - F01 and F10 - and intermediate optical flow Ft1
        along with its relative index.

        Parameters
        ----------
            index : int
                Index

        Returns
        -------
            tuple
                (I0, It, I1, F01, F10, Ft1, name)
        """
        
        if self.train % 3 == 0:
            ### Data Augmentation ###
            # Apply random crop on the 9 input frames
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            reverseFrame = random.randint(0, 1)
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            reverseFrame = 0
            randomFrameFlip = 0
        
        index0 = 0 if not reverseFrame else 2
        index2 = 2 if not reverseFrame else 0
        index3 = 3 if not reverseFrame else 4
        index4 = 4 if not reverseFrame else 3
        
        name = os.path.basename(self.framesPath[index][index0])
        name, _ = os.path.splitext(name)
        
        
        I0 = _pil_loader(self.framesPath[index][index0], resizeDim=self.dim, cropArea=cropArea, frameFlip=randomFrameFlip)
        I1 = _pil_loader(self.framesPath[index][index2], resizeDim=self.dim, cropArea=cropArea, frameFlip=randomFrameFlip)
        
        F01 = _flo_loader(self.framesPath[index][index3], resizeDim=self.dim, cropArea=cropArea, frameFlip=randomFrameFlip)
        F10 = _flo_loader(self.framesPath[index][index4], resizeDim=self.dim, cropArea=cropArea, frameFlip=randomFrameFlip)
        
        if self.train % 3 < 2: 
            if os.path.exists(self.framesPath[index][5]):
                Ft1 = _flo_loader(self.framesPath[index][5], resizeDim=self.dim, cropArea=cropArea, frameFlip=randomFrameFlip)
                Ft1 = -Ft1 if reverseFrame else Ft1
            else:
                u,v = flow_interp(F01, F10, np.array(I0), np.array(I1), 0.5)
                Ft1 = 0.5*np.stack((u,v), axis=-1)
        else:
            Ft1 = F01.copy()
        
        It = Image.new(I0.mode, self.origDim)
        if self.train % 3 < 2:
            It = _pil_loader(self.framesPath[index][1], cropArea=cropArea, frameFlip=randomFrameFlip)
        
        I0 = transforms.ToTensor()(I0)
        It = transforms.ToTensor()(It)
        I1 = transforms.ToTensor()(I1)
        F01 = transforms.ToTensor()(F01)
        F10 = transforms.ToTensor()(F10)
        Ft1 = transforms.ToTensor()(Ft1)
        
        # Apply transformation on images if specified.
        if self.transform is not None:
            I0 = self.transform(I0)
            It = self.transform(It)
            I1 = self.transform(I1)
            
        return I0, It, I1, F01, F10, Ft1, name

    def __len__(self):
        """
        Returns the size of dataset. Invoked as len(datasetObj).

        Returns
        -------
            int
                number of samples.
        """

        return len(self.framesPath)

    def __repr__(self):
        """
        Returns printable representation of the dataset object.

        Returns
        -------
            string
                info.
        """


        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
