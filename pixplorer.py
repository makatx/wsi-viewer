from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import openslide
import numpy as np
from PIL import Image
import cv2
import sys
import importlib
import argparse
import json
import os
from SlideModelRunner import SlideModelRunner
from queue import Queue
from pathlib import Path

class AddNewColumnWorker(QRunnable):
    def __init__(self, tileManager, onLeft, debug=""):
        super().__init__()
        self.tileManager = tileManager
        self.onLeft = onLeft
        self.debug = debug
    
    def run(self):
        #print('{0}: about to run thread'.format(self.debug))
        self.tileManager.mutex.lock()
        self.tileManager.addNewColumn(onLeft=self.onLeft)
        self.tileManager.grid_update_locked = False
        self.tileManager.mutex.unlock()
        #print('{0}: thread complete'.format(self.debug))
        
class AddNewRowWorker(QRunnable):
    def __init__(self, tileManager, onTop, debug=""):
        super().__init__()
        self.tileManager = tileManager
        self.onTop = onTop
        self.debug = debug
    
    def run(self):
        #print('{0}: about to run thread'.format(self.debug))
        self.tileManager.mutex.lock()
        self.tileManager.addNewRow(onTop=self.onTop)
        self.tileManager.grid_update_locked = False
        self.tileManager.mutex.unlock()
        #print('{0}: thread complete'.format(self.debug))
    
class AddNewColumnRowWorker(QRunnable):
    def __init__(self, tileManager, onLeft, onTop, debug=""):
        super().__init__()
        self.tileManager = tileManager
        self.onLeft = onLeft
        self.onTop = onTop
        self.debug = debug
    
    def run(self):
        #print('{0}: about to run thread'.format(self.debug))
        self.tileManager.mutex.lock()
        
        self.tileManager.addNewColumn(onLeft=self.onLeft)
        self.tileManager.addNewRow(onTop=self.onTop)
        
        self.tileManager.grid_update_locked = False
        self.tileManager.mutex.unlock()
        #print('{0}: thread complete'.format(self.debug))

class SlideTile(QGraphicsPixmapItem):
    def __init__(self, pixmap, parent=None, cell=[0,0], slide_coords=[0,0]):
        super().__init__(pixmap)
        self.cell = cell
        self.slide_coords = slide_coords

class TileManager(QObject):
    removeTile = pyqtSignal(SlideTile)
    addTile = pyqtSignal(SlideTile)
    
    def __init__(self, graphicsScene, grid_size=3, tile_size=512, start_level=5, file=None, start_origin=[0,0], start_pos=[0,0]):
        super().__init__()
        
        self.graphicsScene = graphicsScene
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.level = start_level
        self.origin = start_origin
        self.scenePos = start_pos
        self.tile_grid = np.empty((grid_size,grid_size), dtype=object)
        
        self.grid_update_locked = False
        self.mutex = QMutex()

        self.mask_overlay = None
        self.show_overlay_mask = False
        self.mask_level = 4
        self.mask_alpha = 0.3
        
        if file != None:
            self.file = file
            if not self.openSlideFile(file): 
                return
            self.initializeGrid()
            self.addPixmapItems()

    def setFile(self, filename):
        self.file = filename
        if not self.openSlideFile(filename):
            return

        if self.slide != None:
            self.reloadGrid()
        else:
            self.initializeGrid()
            self.addPixmapItems()


    def openSlideFile(self, file):        
        try:
            self.slide = openslide.OpenSlide(file)
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            self.level_count = self.slide.level_count
        except:
            dlg = ErrorPrompt(str(sys.exc_info()))
            dlg.exec_()
            return False
        return True
    
    def setMaskFile(self, file):
        if file==None or not hasattr(self, "file"):
            dlg = ErrorPrompt("A slide file must first be open")
            dlg.exec_()
            return
        try:
            self.mask_file = file
            img = Image.fromarray(np.array(Image.open(file))[:,:,:3])
            self.mask_overlay = img.resize(self.level_dimensions[self.mask_level], Image.NEAREST)
            self.displayMaskOverlay()
        except:
            dlg = ErrorPrompt(str(sys.exc_info()))
            dlg.exec_()


    def displayMaskOverlay(self):
        self.show_overlay_mask = True
        self.reloadGrid()

    def hideMaskOverlay(self):
        self.show_overlay_mask = False
        self.reloadGrid()

    def getTileImage(self, xy):
        '''
        returns image for the tile - may be blended with the image mask if show_overlay_mask is True
        param xy: tuple of (x,y) coordinates to get the image from (given in slide coordinate format - level 0)

        '''
        if not self.show_overlay_mask or self.mask_overlay==None:
            return self.slide.read_region(xy, self.level, (self.tile_size, self.tile_size))
            
        else:
            slide_tile = np.array(self.slide.read_region(xy, self.level, (self.tile_size, self.tile_size)))[:,:,0:3]

            xy_end = xy[0]+self.tile_size*self.level_downsamples[self.level] , xy[1]+self.tile_size*self.level_downsamples[self.level]  
            mask_region_box = np.array([xy[0], xy[1], xy_end[0], xy_end[1]]) // self.level_downsamples[self.mask_level]

            if (mask_region_box[0]<0 and mask_region_box[2]<0) or (mask_region_box[1]<0 and mask_region_box[3]<0):
                return Image.fromarray(slide_tile)
            if (mask_region_box[0]>self.mask_overlay.size[0] and mask_region_box[2]>self.mask_overlay.size[0]) or (mask_region_box[1]>self.mask_overlay.size[1] and mask_region_box[3]>self.mask_overlay.size[1]):
                return Image.fromarray(slide_tile)
            # Padding for out of bound area request
            top_padding, bottom_padding, left_padding, right_padding = 0, 0, 0, 0

            if mask_region_box[0] < 0:
                left_padding = int(-1*mask_region_box[0])
                mask_region_box[0] = 0

            if mask_region_box[1] < 0:
                top_padding = int(-1*mask_region_box[1])
                mask_region_box[1] = 0

            if mask_region_box[2] > self.mask_overlay.size[0]:
                right_padding = int(mask_region_box[2] - self.mask_overlay.size[0])
                mask_region_box[2] = self.mask_overlay.size[0]

            if mask_region_box[3] > self.mask_overlay.size[1]:
                bottom_padding = int(mask_region_box[3] - self.mask_overlay.size[1])
                mask_region_box[3] = self.mask_overlay.size[1]

            mask_region_box = mask_region_box.astype(np.int16)

            try:
                fxy = self.level_downsamples[self.mask_level]/self.level_downsamples[self.level]
                
                ## Alternate implementation (part of) that is slower:
                #mask_region = np.array(self.mask_overlay)[mask_region_box[1]:mask_region_box[3], mask_region_box[0]:mask_region_box[2], :]
                #mask_region = cv2.resize(mask_region, (0,0), fx=fxy, fy=fxy )#, interpolation=cv2.INTER_NEAREST)
                
                width_fxy = int((mask_region_box[2]-mask_region_box[0]) * fxy)
                height_fxy = int((mask_region_box[3]-mask_region_box[1]) * fxy)

                mask_region = np.array(self.mask_overlay.resize((width_fxy, height_fxy), Image.NEAREST, mask_region_box))
               
                if not np.any(mask_region):
                                    return Image.fromarray(slide_tile)

                ## Add padding to make size (tile_size, tile_size)
                if mask_region.shape[1] < self.tile_size:
                    padding_horiz = self.tile_size - mask_region.shape[1]
                    if left_padding > 0:
                        mask_region = np.pad(mask_region, ((0,0), (padding_horiz,0), (0,0)), 'constant')
                    else:
                        mask_region = np.pad(mask_region, ((0,0), (0, padding_horiz), (0,0)), 'constant')
                    
                if mask_region.shape[0] < self.tile_size:
                    padding_vert = self.tile_size - mask_region.shape[0]
                    if top_padding > 0:
                        mask_region = np.pad(mask_region, ((padding_vert,0), (0,0), (0,0)), 'constant')
                    else:
                        mask_region = np.pad(mask_region, ((0, padding_vert), (0,0), (0,0)), 'constant')

                #mask_region = np.pad(mask_region, ((top_padding, bottom_padding), (left_padding, right_padding), (0,0)), 'constant')
                #mask_region = slide_tile

                blended_tile = Image.fromarray(cv2.addWeighted(slide_tile, 1.0, mask_region, self.mask_alpha, 0.0))
            
            except:
                dlg = ErrorPrompt(str(sys.exc_info()))
                dlg.exec_()
                sys.exit(1)

            return blended_tile

    def initializeGrid(self):
        if not hasattr(self, "file"):
            return
        for j in range(self.grid_size):
            for i in range(self.grid_size):
                x = int( self.origin[0]+i*self.tile_size*self.level_downsamples[self.level] )
                y = int( self.origin[1]+j*self.tile_size*self.level_downsamples[self.level] )
                img = self.getTileImage((x,y))
                self.tile_grid[j,i] = SlideTile(img.toqpixmap(), cell=[j,i], slide_coords=[x,y])
    
    def addPixmapItems(self):
        if not hasattr(self, "file"):
            return
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.graphicsScene.addItem(self.tile_grid[i,j])
                self.tile_grid[i,j].setPos(self.scenePos[0]+j*self.tile_size, self.scenePos[1]+i*self.tile_size)
 
    def reloadGrid(self):
        if not hasattr(self, "file"):
            return
        self.removePixmapItems()
        self.initializeGrid()
        self.addPixmapItems()
    
    def updateView(self, level, origin, scenePos):
        self.level = level
        self.origin = origin
        self.scenePos = scenePos
        self.reloadGrid()

    def removePixmapItems(self):
        if not hasattr(self, "file"):
            return
        for item in self.tile_grid.flatten():
            self.graphicsScene.removeItem(item)
    
    
    def addNewColumn(self, onLeft):
        '''
        adds new column of tiles (pixmap items) on left (if onLeft is True else on right) of grid and 
        then adds them to the graphicsScene object at appropriate location
        '''
        
        if onLeft:
            reference_item = self.tile_grid[0,0]
            ref_x = reference_item.x()
            ref_y = reference_item.y()
            slide_coords = reference_item.slide_coords
            
            if slide_coords[0] <= 0:
                ## out of image bounds
                return
            
            x = int(slide_coords[0]-self.tile_size*self.level_downsamples[self.level])
            
            for i in range(self.grid_size):
                y = int( slide_coords[1]+i*self.tile_size*self.level_downsamples[self.level] )
                #print("({0}, {1}), {2}, {3}".format(x,y,self.level, self.tile_size))
                img = self.getTileImage((x,y))
                self.removeTile.emit(self.tile_grid[i,-1])
                self.tile_grid[i,-1] = SlideTile(img.toqpixmap(), cell=[i,0], slide_coords=[x,y])
                self.addTile.emit(self.tile_grid[i,-1])
                self.tile_grid[i,-1].setPos(ref_x-self.tile_size, ref_y+i*self.tile_size)
                
            self.tile_grid = np.roll(self.tile_grid, 1, axis=1)
            self.reindexGrid()
        
        else:
            reference_item = self.tile_grid[0,-1]
            ref_x = reference_item.x()
            ref_y = reference_item.y()
            slide_coords = reference_item.slide_coords
                        
            x = int(slide_coords[0]+self.tile_size*self.level_downsamples[self.level])
            if x >= self.level_dimensions[0][0]:
                ## out of image bounds
                return            
            
            
            for i in range(self.grid_size):
                y = int( slide_coords[1]+i*self.tile_size*self.level_downsamples[self.level] )
                img = self.getTileImage((x,y))
                self.removeTile.emit(self.tile_grid[i,0])
                self.tile_grid[i,0] = SlideTile(img.toqpixmap(), cell=[i,self.tile_size-1], slide_coords=[x,y])
                self.addTile.emit(self.tile_grid[i,0])
                self.tile_grid[i,0].setPos(ref_x+self.tile_size, ref_y+i*self.tile_size)
                
            self.tile_grid = np.roll(self.tile_grid, -1, axis=1)
            self.reindexGrid()
    
    def addNewRow(self, onTop):
        '''
        adds new row of tiles (pixmap items) on top (if onTop is True else at bottom) of grid and 
        then adds them to the graphicsScene object at appropriate location
        '''
        if onTop:
            reference_item = self.tile_grid[0,0]
            ref_x = reference_item.x()
            ref_y = reference_item.y()
            slide_coords = reference_item.slide_coords
            
            if slide_coords[1] <= 0:
                ## out of image bounds
                return            
                        
            y = int(slide_coords[1]-self.tile_size*self.level_downsamples[self.level])
            
            for i in range(self.grid_size):
                x = int( slide_coords[0]+i*self.tile_size*self.level_downsamples[self.level] )
                #print("({0}, {1}), {2}, {3}".format(x,y,self.level, self.tile_size))
                img = self.getTileImage((x,y))
                self.removeTile.emit(self.tile_grid[-1,i])
                self.tile_grid[-1,i] = SlideTile(img.toqpixmap(), cell=[0,i], slide_coords=[x,y])
                self.addTile.emit(self.tile_grid[-1,i])
                self.tile_grid[-1,i].setPos(ref_x+i*self.tile_size, ref_y-self.tile_size)
                
            self.tile_grid = np.roll(self.tile_grid, 1, axis=0)
            self.reindexGrid()
        
        else:
            reference_item = self.tile_grid[-1,0]
            ref_x = reference_item.x()
            ref_y = reference_item.y()
            slide_coords = reference_item.slide_coords
         
            y = int(slide_coords[1]+self.tile_size*self.level_downsamples[self.level])
            if y >= self.level_dimensions[0][1]:
                ## out of image bounds
                return            

            
            
            for i in range(self.grid_size):
                x = int( slide_coords[0]+i*self.tile_size*self.level_downsamples[self.level] )
                img = self.getTileImage((x,y))
                self.removeTile.emit(self.tile_grid[0,i])
                self.tile_grid[0,i] = SlideTile(img.toqpixmap(), cell=[self.tile_size-1,i], slide_coords=[x,y])
                self.addTile.emit(self.tile_grid[0,i])
                self.tile_grid[0,i].setPos(ref_x+i*self.tile_size, ref_y+self.tile_size)
                
            self.tile_grid = np.roll(self.tile_grid, -1, axis=0)
            self.reindexGrid()
    
    
    def reindexGrid(self):
        for i in range(self.tile_grid.shape[0]):
            for j in range(self.tile_grid.shape[1]):
                self.tile_grid[i,j].cell = [i,j]
        
        
    def zoomIn(self, zoomPoint, itemCell, tilePos):
        level = self.level-1
        if level < 0 :
            return
        
        focused_tile = self.tile_grid[itemCell[0], itemCell[1]]
        focused_slide_coords = focused_tile.slide_coords
        focused_slide_point = [ focused_slide_coords[0]+tilePos[0]*self.level_downsamples[self.level], \
            focused_slide_coords[1]+tilePos[1]*self.level_downsamples[self.level] ]
        
        new_origin = [ int(focused_slide_point[0]-self.grid_size*self.tile_size*self.level_downsamples[level]/2),\
            int(focused_slide_point[1]-self.grid_size*self.tile_size*self.level_downsamples[level]/2) ]
    
        new_scenePos = [ (zoomPoint[0]-self.grid_size*self.tile_size/2), \
            (zoomPoint[1]-self.grid_size*self.tile_size/2) ]
        
        self.updateView(level, new_origin, new_scenePos) 
        
    
    
    def zoomOut(self, zoomPoint, itemCell, tilePos):
        level = self.level+1
        
        if level >= self.level_count:
            return
            
        focused_tile = self.tile_grid[itemCell[0], itemCell[1]]
        focused_slide_coords = focused_tile.slide_coords
        focused_slide_point = [ focused_slide_coords[0]+tilePos[0]*self.level_downsamples[self.level],\
            focused_slide_coords[1]+tilePos[1]*self.level_downsamples[self.level] ]
        
        new_origin = [ int(focused_slide_point[0]-self.grid_size*self.tile_size*self.level_downsamples[level]/2),\
            int(focused_slide_point[1]-self.grid_size*self.tile_size*self.level_downsamples[level]/2) ]
    
        new_scenePos = [ (zoomPoint[0]-self.grid_size*self.tile_size/2),\
            (zoomPoint[1]-self.grid_size*self.tile_size/2) ]
        
        self.updateView(level, new_origin, new_scenePos) 
        
        
class Pixplorer(QGraphicsScene):

    def __init__(self, file, parent=None,  grid_size=3, tile_size=512, start_level=5, start_origin=[0,0], window_size=(512,512)):
        super().__init__(parent)
        
        ## pan speed adjustment
        self.lam = 2 
        
        ## [length, height]
        self.window_size = window_size 

        self.grid_size = grid_size
        self.tile_size = tile_size
        
        self.last_diff_x = np.zeros((10), dtype=np.int16)
        self.last_diff_y = np.zeros((10), dtype=np.int16)

        scene_start = [ ((grid_size*tile_size-window_size[0])/2), ((grid_size*tile_size-window_size[1])/2) ]  
        self.setSceneRect(scene_start[0], scene_start[1], window_size[0], window_size[1])
        
        self.tileManager = TileManager(self, file=file, grid_size=grid_size, tile_size=tile_size, start_level=start_level, start_origin=start_origin)
        self.tileManager.addTile.connect(self.slotAddTile)
        self.tileManager.removeTile.connect(self.slotRemoveTile)

        self.threadPool = QThreadPool()
        #self.threadPool.setMaxThreadCount(2)

    @pyqtSlot(SlideTile)
    def slotRemoveTile(self, SlideTile):
        self.removeItem(SlideTile)

    @pyqtSlot(SlideTile)
    def slotAddTile(self, SlideTile):
        self.addItem(SlideTile)

    def mouseMoveEvent(self, event):
                       
        p = event.lastScenePos()
        x, y = p.x(), p.y()
        
        p = event.scenePos()
        nx, ny = p.x(), p.y()
        
        diff_x = nx-x
        diff_y = ny-y
                           
        scene_rect = self.sceneRect()
        
        scene_x = scene_rect.x()
        scene_y = scene_rect.y()
        
        if np.average(self.last_diff_x) * diff_x > 0:
            new_scene_x = scene_x - self.lam*diff_x
        else:
            new_scene_x = scene_x
            
        if np.average(self.last_diff_y) * diff_y > 0:
            new_scene_y = scene_y - self.lam*diff_y
        else:
            new_scene_y = scene_y
            
        self.last_diff_x = np.roll(self.last_diff_x, 1)
        self.last_diff_x[0] = diff_x
        self.last_diff_y = np.roll(self.last_diff_y,1)
        self.last_diff_y[0] = diff_y
        
        new_view_centre = [new_scene_x+self.window_size[0]/2, new_scene_y+self.window_size[1]]
        new_centre_item = self.itemAt(new_view_centre[0], new_view_centre[1], QTransform())
        
        if (type(new_centre_item)==type(SlideTile(QPixmap()))):

            self.setSceneRect(new_scene_x, new_scene_y, self.window_size[0], self.window_size[1])
            
            if self.tileManager.grid_update_locked:
                return

            centre_cell_index = new_centre_item.cell
            move_diff = [self.grid_size//2-centre_cell_index[0], self.grid_size//2-centre_cell_index[1]]

            if move_diff[1] > 0 and move_diff[0]==0:
                #add column to left
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnWorker(self.tileManager, onLeft=True, debug='left only'))
                
            elif move_diff[1] < 0 and move_diff[0]==0:
                #add column to right
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnWorker(self.tileManager, onLeft=False, debug='right only'))
                
            elif move_diff[0] > 0 and move_diff[1]==0:
                #add row on top
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewRowWorker(self.tileManager, onTop=True, debug='top only'))
                
            elif move_diff[0] < 0 and move_diff[1]==0:
                #add row at bottom
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewRowWorker(self.tileManager, onTop=False, debug='bottom only'))
                
            elif move_diff[1] > 0 and move_diff[0] > 0 :
                #add column to left and then row on top
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnRowWorker(self.tileManager, onLeft=True, onTop=True, debug='left, top'))
                #self.threadPool.start(AddNewRowWorker(self.tileManager, debug='left, now top'))
                
            elif move_diff[1] > 0 and move_diff[0] < 0 :
                #add column to left and then row on bottom
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnRowWorker(self.tileManager, onLeft=True, onTop=False, debug='left, bottom'))
                #self.threadPool.start(AddNewRowWorker(self.tileManager, onTop=False, debug='left, now bottom'))
                
            elif move_diff[1] < 0 and move_diff[0] > 0 :
                #add column to right and then row on top
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnRowWorker(self.tileManager, onLeft=False, onTop=True, debug='right, top'))
                #self.threadPool.start(AddNewRowWorker(self.tileManager, onTop=True, debug='right, now top'))
                
            elif move_diff[1] < 0 and move_diff[0] < 0 :
                #add column to right and then row on bottom
                self.tileManager.grid_update_locked = True
                self.threadPool.start(AddNewColumnRowWorker(self.tileManager, onLeft=False, onTop=False, debug='right, bottom'))
                #self.threadPool.start(AddNewRowWorker(self.tileManager, onTop=False, debug='right, now bottom'))
                
    def wheelEvent(self, event):
        delta = event.delta()
        cursorScenePos = [event.scenePos().x(), event.scenePos().y()]
        #pos = [event.pos().x(), event.pos().y()]
        item = self.itemAt(cursorScenePos[0], cursorScenePos[1], QTransform())

        if (type(item)==type(SlideTile(QPixmap()))):
            cursorTilePos = item.mapFromParent(cursorScenePos[0], cursorScenePos[1])
                        
            if delta>0:
                self.tileManager.zoomIn(cursorScenePos, item.cell, [cursorTilePos.x(), cursorTilePos.y()])
                #self.updateSceneRect()
            elif delta<0:
                self.tileManager.zoomOut(cursorScenePos, item.cell, [cursorTilePos.x(), cursorTilePos.y()])
                #self.updateSceneRect()
        
    def centerSceneRect(self):
        n = self.grid_size
        t = self.tile_size
        w = self.window_size
        x_s = self.tileManager.scenePos[0]
        y_s = self.tileManager.scenePos[1]

        x_rect = x_s + (n*t-w[0])/2
        y_rect = y_s + (n*t-w[1])/2

        self.setSceneRect(x_rect, y_rect, w[0], w[1])

class ModelSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Select and configure a Keras/TF model to run on the current slide")
        self.formFields = {}
        
        self.vlayout = QVBoxLayout()
        self.formLayout = QFormLayout()
        
        self.model_file_selection_gb = QGroupBox('Select Model File')
        self.model_file_vbox = QVBoxLayout()
        self.model_file_text = QLineEdit()
        self.model_file_browse = QPushButton('Browse...')
        self.model_file_browse.clicked.connect(self.populate_model_file)
        self.model_file_vbox.addWidget(self.model_file_text)
        self.model_file_vbox.addWidget(self.model_file_browse)
        self.model_file_selection_gb.setLayout(self.model_file_vbox)
        

        self.masksave_file_selection_gb = QGroupBox('Select Output Mask Save Location and Prefix')
        self.masksave_file_vbox = QVBoxLayout()
        self.masksave_file_text = QLineEdit()
        self.masksave_file_browse = QPushButton('Browse...')
        self.masksave_file_browse.clicked.connect(self.populate_save_file)
        self.masksave_file_vbox.addWidget(self.masksave_file_text)
        self.masksave_file_vbox.addWidget(self.masksave_file_browse)
        self.masksave_file_selection_gb.setLayout(self.masksave_file_vbox)

        self.parameters_gb = QGroupBox('Parameters')

        self.tile_size_spinbox = QSpinBox(self)
        self.tile_size_spinbox.setMinimum(128)
        self.tile_size_spinbox.setMaximum(512)
        self.tile_size_spinbox.setSingleStep(128)
        self.tile_size_spinbox.setValue(256)
        self.formLayout.addRow(QLabel("Tile Size"), self.tile_size_spinbox)

        self.batch_size_spinbox = QSpinBox(self)
        self.batch_size_spinbox.setMinimum(32)
        self.batch_size_spinbox.setMaximum(128)
        self.batch_size_spinbox.setSingleStep(16)
        self.batch_size_spinbox.setValue(64)
        self.formLayout.addRow(QLabel("Batch Size"), self.batch_size_spinbox)

        self.prediction_level_spinbox = QSpinBox(self)
        self.prediction_level_spinbox.setMinimum(0)
        self.prediction_level_spinbox.setMaximum(3)
        self.prediction_level_spinbox.setSingleStep(1)
        self.prediction_level_spinbox.setValue(1)
        self.formLayout.addRow(QLabel("Prediction Level"), self.prediction_level_spinbox)

        self.parameters_gb.setLayout(self.formLayout)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.validateAccept)
        self.buttonBox.rejected.connect(self.reject)


        self.vlayout.addWidget(self.model_file_selection_gb)
        self.vlayout.addWidget(self.masksave_file_selection_gb)
        self.vlayout.addWidget(self.parameters_gb)
        self.vlayout.addWidget(self.buttonBox)

        self.setLayout(self.vlayout)

    def populate_model_file(self):
        filename = QFileDialog.getOpenFileName(self, "Select Model File", "./", "Model Saves (*.h5)")
        if filename[0] != '':
            self.model_file_text.setText(filename[0])

    def populate_save_file(self):
        filename = QFileDialog.getSaveFileName(self, "Save Output Mask File To...", "./", "Image (*.png)")
        if filename[0] != '':
            self.masksave_file_text.setText(filename[0])

    def validateAccept(self):
        if self.model_file_text.text() == '' or not os.path.exists(self.model_file_text.text()):
            ErrorPrompt("Please select a valid model file to load.", "Input needed").exec()
            return
        if self.masksave_file_text.text() == '':
            ErrorPrompt("Output mask will be saved to same location as the slide file", "Warning").exec()


        self.formFields['model_file'] = self.model_file_text.text()
        self.formFields['masksave_file'] = self.masksave_file_text.text()
        self.formFields['tile_size'] = self.tile_size_spinbox.value()
        self.formFields['batch_size'] = self.batch_size_spinbox.value()
        self.formFields['prediction_level'] = self.prediction_level_spinbox.value()

        self.accept()

class ProgressDialog(QDialog):
    def __init__(self, modeloptions=None, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Running selected model...")
        self.layout = QVBoxLayout()

        self.label = QLabel("")
        self.layout.addWidget(self.label)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.layout.addWidget(self.progressBar)

        self.setLayout(self.layout)
        if modeloptions != None:
            self.setLabelText(modeloptions)

    @pyqtSlot(str, int)
    def updateProgress(self, msg, value):
        self.progressBar.setValue(value)
        if value >= 100:
            self.close()

    def setLabelText(self, modeloptions):
        modelinfo = "Running model\t:\t{}\n \
            On slide\t:\t{}\n".format(modeloptions['model_file'], modeloptions['slide_file'])
        self.label.setText(modelinfo)

class ModelRunThread(QRunnable):
    def __init__(self, slideModelRunner, resultQueue):
        super().__init__()
        self.slideModelRunner = slideModelRunner
        self.resultQueue = resultQueue

    def run(self):
        self.resultQueue.put(self.slideModelRunner.evaluateModelOnSlide())
        self.slideModelRunner.modelRunCompleteSignal.emit()

class MainWindow(QMainWindow):
    '''
    im_plugin_file is the full path to the json file listing the image processing plugins
    '''
    OPENSLIDE_FORMATS = "All OpenSlide Supported Formats (*.svs *.tif *.ndpi *.vms *.vmu *.scn *.mrxs *.tiff *.svslide *.tif *.bif) ;; \
        Aperio (*.svs *.tif) ;; \
        Hamamatsu (*.ndpi *.vms *.vmu) ;; \
        Leica (*.scn) ;; \
        MIRAX (*.mrxs) ;; \
        Philips (*.tiff) ;; \
        Sakura (*.svslide) ;; \
        Trestle (*.tif) ;; \
        Ventana (*.bif *.tif) ;; \
        Generic tiled TIFF (*.tif)"

    unloadModelSignal = pyqtSignal()

    def __init__(self, im_plugin_file=None):
        super().__init__()

        #TODO: Make the application start backgroud dark
        #plt = QGuiApplication.palette()
        #self.setAutoFillBackground(True)

        self.modelRunner = None
        self.file = None

        self.setWindowTitle("Path-Pixplorer")
        geometry  = qApp.desktop().availableGeometry()
        self.setGeometry(geometry)
        self.showMaximized()

        self.openMenu = self.menuBar().addMenu("Open")

        openAction = QAction("Slide", self)
        openAction.triggered.connect(self.openSlideFile)
        self.openMenu.addAction(openAction)

        openMaskAction = QAction("Mask", self)
        openMaskAction.triggered.connect(self.openMask)
        self.openMenu.addAction(openMaskAction)

        closeAction = QAction("Exit", self)
        closeAction.triggered.connect(self.close)
        self.openMenu.addAction(closeAction)

        self.modelMenu = self.menuBar().addMenu("Model...")

        self.modelSelector = ModelSelectDialog(parent=self)
        self.selectModel = QAction("Select Model", self)
        self.selectModel.setEnabled(False)
        self.selectModel.triggered.connect(self.setModelFile)
        self.modelMenu.addAction(self.selectModel)

        self.runModelAction = QAction("Run model on current slide")
        self.runModelAction.setEnabled(False)
        self.runModelAction.triggered.connect(self.runSelectedModel)
        self.modelMenu.addAction(self.runModelAction)

        self.toggleMaskAction = QAction("Toggle Mask", self)
        self.toggleMaskAction.triggered.connect(self.toggleMask)
        self.toggleMaskAction.setEnabled(False)
        self.menuBar().addAction(self.toggleMaskAction)

        self.graphicsScene = Pixplorer(file=None, grid_size=5, start_origin=(28807, 138807), start_level=3, window_size=(768,768))
        self.graphicsView = QGraphicsView(self.graphicsScene)
        self.setCentralWidget(self.graphicsView)

        if im_plugin_file != None:
            self.pluginMenu = self.menuBar().addMenu("Plugins")
            self.plugins = []
            try:
                self.loadPlugins(im_plugin_file)
            except:
                dlg = ErrorPrompt("Could not load plugins:\n"+str(sys.exc_info()))
                dlg.exec_()

        self.progressBar = None

        self.progressDialog = ProgressDialog(parent=self)

    def openSlideFile(self):
        dialog = QFileDialog(self, "Select slide file to open", "/home/mak/PathAI/slides")
        dialog.setNameFilter(self.OPENSLIDE_FORMATS)

        if dialog.exec_():
            filename = dialog.selectedFiles()
            if self.file != filename[0]:
                self.file = filename[0]
                self.graphicsScene.tileManager.setFile(self.file)
                self.selectModel.setEnabled(True)
                if self.modelRunner != None:
                    self.modelRunner.updateSlideFile(self.file)

    def openMask(self):
        filename = QFileDialog.getOpenFileName(self, "Select Mask File", "", "Image (*.png *.jpg *.jpeg *.bmp)")
        if filename[0] != '':
            self.mask_file = filename[0]
            self.graphicsScene.tileManager.setMaskFile(filename[0])
            self.toggleMaskAction.setEnabled(True)

    def setModelFile(self):
        #TODO: add slot for when the model file selection changes
        if self.modelSelector.exec() == QDialog.Accepted:
            self.currentModelOptions = self.modelSelector.formFields
            self.currentModelOptions['slide_file'] = self.file
            self.runModelAction.setEnabled(True)
            if self.modelRunner != None:
                self.modelRunner.updateParameters(self.currentModelOptions)

    def runSelectedModel(self):
        if self.modelRunner == None:
            opts = self.currentModelOptions
            self.modelRunner = SlideModelRunner(opts['model_file'], self.file, tile_size=opts['tile_size'], batch_size=opts['batch_size'], prediction_level=opts['prediction_level'])
            self.modelRunner.modelRunCompleteSignal.connect(self.modelRunComplete)
            self.modelRunner.progressSignal.connect(self.progressDialog.updateProgress)
            self.unloadModelSignal.connect(self.modelRunner.unloadModel)
        
        self.progressDialog.setLabelText(self.currentModelOptions)
        self.resultQueue = Queue()
        modelRunWorker = ModelRunThread(self.modelRunner, self.resultQueue)
        self.threadPool = QThreadPool()
        self.threadPool.start(modelRunWorker)
        self.progressDialog.exec()

        #mask_img = self.modelRunner.getMaskFromPredictions()
        #Image.fromarray(mask_img).save('mask.png')

    @pyqtSlot()
    def modelRunComplete(self):
        print('Generating mask from predictions... ', self.modelRunner.predictions.shape)
        mask_save_file = self.currentModelOptions['masksave_file']
        if mask_save_file == '':
            mask_save_file_folder = Path(self.file).parent
            mask_save_file_prefix = 'MASK'
        else:
            mask_save_file_folder = Path(mask_save_file).parent
            mask_save_file_prefix = Path(mask_save_file).stem
        mask_save_file_path = mask_save_file_folder / (mask_save_file_prefix + Path(self.file).stem + '.png')

        predictions_mask = self.modelRunner.getMaskFromPredictions()
        Image.fromarray(predictions_mask).save(mask_save_file_path)
        print('Saved mask to ', mask_save_file_path)

        self.graphicsScene.tileManager.setMaskFile(mask_save_file_path)

    def loadPlugins(self, file):
        with open(file, 'r') as f:
            plg_list = json.load(f)['ImageProcessingPlugins']
        for plg in plg_list:
            mod = plg['directory']+"."+plg['module']
            class_name = plg['class_name']
            self.loadPluginActions(mod, class_name)

    def loadPluginActions(self, mod, class_name):
        module = importlib.import_module(mod)
        plugin_class = getattr(module, class_name)
        plugin_object = plugin_class(self.graphicsScene.tileManager)
        plugin_object.progressSignal.connect(self.updateProgress)

        plugin_action = QAction(plugin_object.action_name, self)
        plugin_action.setToolTip(plugin_object.tooltip)
        plugin_action.triggered.connect(plugin_object.runAction)

        self.pluginMenu.addAction(plugin_action)
        self.plugins.append(plugin_object)
        
    @pyqtSlot(str, int)
    def updateProgress(self, mesg, value):
        if self.progressBar == None:
             self.progressBar = QProgressBar()
             self.progressBar.setMinimum(0)
             self.progressBar.setMaximum(100)
             self.progressBar.setValue(0)
             self.statusBar().addWidget(self.progressBar)
        if value >= 100:
            self.statusBar().removeWidget(self.progressBar)
            self.progressBar = None
        else:
            self.progressBar.setValue(value)
            self.statusBar().showMessage(mesg)

    def toggleMask(self):
        if self.graphicsScene.tileManager.show_overlay_mask:
            self.graphicsScene.tileManager.hideMaskOverlay()
        else:
            self.graphicsScene.tileManager.displayMaskOverlay()

class ErrorPrompt(QDialog):

    def __init__(self, message, title="An Error Occured"):
        super(ErrorPrompt, self).__init__()
        
        self.setWindowTitle(title)
        
        QBtn = QDialogButtonBox.Ok
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.close)

        self.label = QLabel(message, parent=self)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


if __name__ == "__main__":

    ## Parse input arguments from commandline
    aparser = argparse.ArgumentParser('Pathology digital whole slide image viewer using OpenSlide')
    aparser.add_argument('--im-plugin', type=str, help='location of the plugins (json) file to load')

    args = aparser.parse_args()

    app = QApplication([])

    window = MainWindow(im_plugin_file=args.im_plugin)

    window.show()
    app.exec_()