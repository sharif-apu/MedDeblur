import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
import imgaug.augmenters as iaa
import random


def motion_blur(image, degreeMax=45, angleMax=45):
    image = np.array(image)
    degree = random.randint(20, degreeMax)
    angle = random.randint(-45, angleMax)
    # This generates a matrix of motion blur kernels at any angle. The greater the degree, the higher the blur.
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    #cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    #blurred = np.array(blurred, dtype=np.uint8)
    return blurred


class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        self.var= 0.1
        self.pov = 0.6
        self.mean = 0.0
        self.normalize = transforms.Normalize(normMean, normStd)

 

       
        self.transformCV = transforms.Compose([ #transforms.Resize((int(self.imageH),int(self.imageW)), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                #AddGaussianNoise(pov=1),
                                                self.normalize#self.normalize
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):

        # Read Images
        
        try:    
            self.gtImage  = cv2.imread(self.image_list[i])#cv2.imread(self.image_list[i ])#Image.open(self.gtImageFileName)#Image.open(self.image_list[i])
        except:
            self.sampledImage = Image.open(self.image_list[i + 1])
            os.remove(i)
            print ("File deleted:", i)
            i += 1

        self.sampledImage= motion_blur(self.gtImage)


        self.inputL1 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL1 = cv2.resize(self.inputL1, (128//4, 128//4)).astype(np.float32)
        self.inputL1 = self.transformCV(self.inputL1)


        self.inputL2 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL2 = cv2.resize(self.inputL2, (128//2, 128//2)).astype(np.float32)
        self.inputL2 = self.transformCV(self.inputL2)


        self.gtImageHR3 = cv2.cvtColor(self.gtImage, cv2.COLOR_BGR2RGB)/255.0
        self.gtImageHR3 = cv2.resize(self.gtImageHR3, (128, 128)).astype(np.float32)
        self.gtImageHR3 = self.transformCV(self.gtImageHR3)

        self.inputL3 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL3 = cv2.resize(self.inputL3, (128, 128)).astype(np.float32)
        self.inputL3 = self.transformCV(self.inputL3)

        return self.inputL1, self.inputL2, self.inputL3, self.gtImageHR3

