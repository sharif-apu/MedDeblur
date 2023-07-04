import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os 
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
import skimage.io as io

class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/255
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.gridSize = gridSize
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
    


    def inputForInference(self, imagePath, noiseLevel):

        img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)/255.0#cv2.imread(imagePath)#Image.open(imagePath) #io.imread(imagePath)/255#
        #print(imagePath)
        #resizeDimension =  (1024, 1024) 
        #img = img.resize(resizeDimension)
        #img.save(imagePath)  

        '''img = np.asarray(img) 
        if self.gridSize == 1 : 
            img = bayerSampler(img)
        elif self.gridSize == 2 : 
            img = quadBayerSampler(img)
        elif self.gridSize == 3 : 
            img = dynamicBayerSampler(img, gridSze)
        img = Image.fromarray(img)'''
        #print("img", img.getextrema())
        normalize = transforms.Normalize(normMean, normStd)
        #width, height = img.size
        if self.resize:
            #resize(256,256)
            transform = transforms.Compose([ transforms.Resize(self.resize, interpolation=Image.BICUBIC) ])
            img = transform(img)


        transformCV = transforms.Compose([ #transforms.Resize((int(width/4),int(height/4)), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize
                                                 ])
        '''transformL2 = transforms.Compose([ transforms.Resize((int(width/2),int(height/2)), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize
                                                 ])
        transformL3 = transforms.Compose([ transforms.ToTensor(),
                                                normalize
                                            ])'''

        '''self.inputL1 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL1 = cv2.resize(self.inputL1, (128//4, 128//4)).astype(np.float32)
        self.inputL1 = self.transformCV(self.inputL1)


        #self.gtImageHR2 = cv2.cvtColor(self.gtImage, cv2.COLOR_BGR2RGB)/255.0
        #self.gtImageHR2 = cv2.resize(self.gtImageHR2, (128//2, 128//2)).astype(np.float32)
        #self.gtImageHR2 = self.transformCV(self.gtImageHR2)

        self.inputL2 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL2 = cv2.resize(self.inputL2, (128//2, 128//2)).astype(np.float32)
        self.inputL2 = self.transformCV(self.inputL2)


        self.gtImageHR3 = cv2.cvtColor(self.gtImage, cv2.COLOR_BGR2RGB)/255.0
        self.gtImageHR3 = cv2.resize(self.gtImageHR3, (128, 128)).astype(np.float32)
        self.gtImageHR3 = self.transformCV(self.gtImageHR3)

        self.inputL3 = cv2.cvtColor(self.sampledImage, cv2.COLOR_BGR2RGB)/255.0
        self.inputL3 = cv2.resize(self.inputL3, (128, 128)).astype(np.float32)
        self.inputL3 = self.transformCV(self.inputL3)'''

        #testImg = cv
        testImgL1 = cv2.resize(img, (img.shape[0]//4, img.shape[1]//4)).astype(np.float32)
        testImgL1 = transformCV(testImgL1).unsqueeze(0)

        testImgL2 = cv2.resize(img, (img.shape[0]//2, img.shape[1]//2)).astype(np.float32)
        testImgL2 = transformCV(testImgL2).unsqueeze(0)
        testImgL3 = transformCV(img.astype(np.float32)).unsqueeze(0)

        #print("input",imagePath,self.unNormalize(testImg).max(), self.unNormalize(testImg).min())
        return testImgL1, testImgL2, testImgL3 

    def saveModelOutput(self, modelOutput, inputImagePath, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                              "_" + self.modelName + "_" + str(step) + ext
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
            "_" + self.modelName + ext
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)
        #print(imageSavingPath)
        #print(inputImagePath,self.unNormalize(modelOutput[0]).max(), self.unNormalize(modelOutput[0]).min())

    

    def testingSetProcessor(self):
        testSets = glob.glob(self.inputRootDir+"*/")
        #print ("DirPath",self.inputRootDir+"*/")
        if self.validation:
            #print(self.validation)
            testSets = testSets[:1]
        #print (testSets)
        testImageList = []
        for t in testSets:
            testSetName = t.split("/")[-2]
            #print("Dir Path",self.outputRootDir + self.modelName  + "/" + testSetName )
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir

        return testImageList


