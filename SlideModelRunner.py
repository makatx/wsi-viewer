from keras.models import Model, load_model
import openslide
from math import ceil
import numpy as np
from skimage.color import rgb2hed
import cv2
from PyQt5.QtCore import QObject, pyqtSignal

class SlideModelRunner(QObject):
    '''
    Takes in a Keras model file and slide file (openslide) with parameters and then loads the model into memory,
    and runs it on the tiles generated from specific given level in the slide.
    Provides utility functions to further process the results and return image masks etc. 
    '''
    progressSignal = pyqtSignal(str, int)
    modelRunCompleteSignal = pyqtSignal()

    def __init__(self, model_file, slide_file, tile_size=256, batch_size=64, ROI_extraction_method='hed_deconv', ROI_level=7, prediction_level=1):
        super().__init__()
        self.model_file = model_file
        self.slide_file = slide_file
        self.slide = self.getWSI(self.slide_file)
        self.tile_size =  tile_size
        self.batch_size = batch_size
        self.ROI_extraction_method = ROI_extraction_method
        self.ROI_level = ROI_level
        self.prediction_level = prediction_level
        self.tile_list = self.getPatchCoordListFromFile(from_level=self.ROI_level)
        self.predictions = None
        
        print("loading model from file: "+self.model_file)
        self.model = load_model(self.model_file)

    def loadModel(self):
        print("loading model from file: "+self.model_file)
        self.model = load_model(self.model_file)

    def patch_batch_generator(self, progressSignal=None):
        '''
        Generating image batches from given slide and level using coordinates in tile_list
        images are normalized: (x-128)/128 before being returned
        '''
        level = self.prediction_level
        dims = [self.tile_size, self.tile_size]
        images = []
        b = 0
        imgs_yielded = 0
        for coord in self.tile_list:
            if b==self.batch_size:
                imgs_yielded += b
                b=0
                images_batch = np.array(images)
                images = []
                if progressSignal != None:
                    progressSignal.emit("Processing tile batches...", ceil(imgs_yielded*100/len(self.tile_list)))
                yield images_batch
            images.append(((self.getRegionFromSlide(level, coord, dims=dims).astype(np.float))-128)/128)
            b +=1
        images_batch = np.array(images)
        if progressSignal != None:
            progressSignal.emit("Processing tile batches...", 100)
        yield images_batch

    def evaluateModelOnSlide(self, progressSignal=None):
        steps = ceil((self.tile_list.shape[0])/self.batch_size)
        gen = self.patch_batch_generator(progressSignal=progressSignal)
        self.predictions = self.model.predict_generator(gen, steps, verbose=1)
        return self.predictions

    def getPatchCoordListFromFile(self, from_level='max'):
            if from_level =='max':
                from_level = self.slide.level_count-1
            img = self.getRegionFromSlide(level=from_level)
            mask = self.getDABMask(img)

            nzs = np.argwhere(mask)
            nzs = nzs * self.slide.level_downsamples[from_level]
            nzs = nzs.astype(np.int32)
            nzs = np.flip(nzs, 1)
            
            return nzs

    def getRegionFromSlide(self, level=8, start_coord=(0,0), dims='full'):
        if dims == 'full':
            img = np.array(self.slide.read_region((0,0), level, self.slide.level_dimensions[level]))
            img = img[:,:,:3]
        else:
            img = np.array(self.slide.read_region(start_coord, level, dims ))
            img = img[:,:,:3]
        return img

    def getWSI(self, filename):
        '''
        Returns OpenSlide object for slide filename
        '''
        slide = openslide.OpenSlide(filename)
        return slide

    def getThresholdMask(self, img, threshold=(140,210), channel=0, margins=None):
        '''
        Retuns threhold applied image for given threhold and channel, suppressing any pixels to 0 for given margins
        params:
        margins: (left_y, right_y, top_x, bottom_x) ;  can be specified as negative as well. ex: (50, -50, 50, -50)
        '''
        mask = np.zeros_like(img[:,:,channel], dtype=np.uint8)
        mask[((img[:,:,channel] > threshold[0]) & (img[:,:,channel] < threshold[1]))] = 255

        if margins != None :
            mask[:margins[0]] = 0
            mask[margins[1]:] = 0

            mask[:, :margins[2]] = 0
            mask[:, margins[3]:] = 0

        return mask

    def getDABThresholdMask(self, hed_img, threshold=(30,150), margins=(50, -50, 50, -50)):
        return self.getThresholdMask(hed_img, threshold, channel=2, margins=margins)

    def performClose(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def performOpen(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def getHED(self, img):
        '''
        Return a channel scaled image in the HED, color deconvolution performed format
        '''
        hed = rgb2hed(img)
        #hed_sc = np.zeros_like(hed)
        for i in range(3):
            r_min = np.min(hed[:,:,i])
            r_max = np.max(hed[:,:,i])
            r = r_max - r_min
            hed[:,:,i] = (hed[:,:,i]-r_min) * 255.0/r
        return hed.astype(np.uint8)

    def getDABMask(self, img):
        '''
        Returns the bit mask for ROI from the given RGB img, using DAB channel of the HED converted image
        '''
        h,w = img.shape[:2]
        margins = [ceil(0.025*h), ceil(-0.025*h), ceil(0.05*w), ceil(-0.05*w)]

        hed = self.getHED(img)
        mask = self.getDABThresholdMask(hed, margins=margins)
        mask = self.performOpen(self.performClose(mask))

        return mask

    def getMaskFromPredictions(self, img_level=7, threshold=0.4, color=[255,255,0]):
        
        img_shape = (self.slide.level_dimensions[img_level][1], self.slide.level_dimensions[img_level][0], 3)
        img = np.zeros(img_shape, dtype=np.uint8)
        scale_multiplier = self.slide.level_downsamples[self.prediction_level]/self.slide.level_downsamples[img_level]
        downsample = self.slide.level_downsamples[img_level]
        tile = ceil(self.tile_size * scale_multiplier)
        for i in range(len(self.tile_list)):
            if self.predictions[i,1] >= threshold:
                coord = ( int(self.tile_list[i][0]/downsample) , int(self.tile_list[i][1]/downsample) )
                img[coord[1]:coord[1]+tile, coord[0]:coord[0]+tile] = color
        self.mask = img
        return img
    