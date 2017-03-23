# -*- coding: utf-8 -*-
"""
Simple example of loading UI template created with Qt Designer.

This example uses uic.loadUiType to parse and load the ui at runtime. It is also
possible to pre-compile the .ui file using pyuic (see VideoSpeedTest and 
ScatterPlotSpeedTest examples; these .ui files have been compiled with the
tools/rebuildUi.py script).
"""
#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui#QStringList,QString
import numpy as np
import os

pg.mkQApp()

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'imagingAnalysis.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

import tifffile
import numpy as np

import db_access as dba
fly_db = dba.get_db()

default_rframe_data = {'a1': np.array([ 51.5848967 ,  -5.93928407]),
                       'a2': np.array([ -0.09151179,  88.42505672]),
                       'p': np.array([ 26.66908747,  34.43488385])}

stacked_muscles = tifffile.TiffFile('stacked_muscles.tiff')
overlay = np.transpose(stacked_muscles.asarray(),(1,0,2))[:,::-1].astype(np.float32)
#tiff_file = '/volumes/FlyDataB/FlyDB/Fly0212/T2_trial1_ND_04_1ms_exposure/T2_trial1_ND_04_1ms_exposure_MMStack.ome.tif'

#tiff_file_name = '/media/FlyDataB/FlyDB/Fly0267/T2_trial1_ND_04_100us_exposure_td_refstack/T2_trial1_ND_04_100us_exposure_td_refstack_MMStack.ome.tif'


class Basis(dict):    
    def __setitem__(self,key,item):
        try:
            if key in ['a1','a2']:
                dict.__setitem__(self,key,item)
                A = np.vstack((self['a1'],self['a2'])).T
                A_inv = np.linalg.inv(A)
                self['A'] = A
                self['A_inv'] = A_inv
            else:
                dict.__setitem__(self,key,item)
        except KeyError:
            dict.__setitem__(self,key,item)
                        
class GeometricModel(object):   
    def __init__(self,lines,basis):
        self.lines = lines
        self.basis = basis
        ## put lines in barycentric coords
        self.barycentric = dict()
        for key in self.lines.keys():
            coords = np.dot(self.basis['A_inv'],(self.lines[key]-self.basis['p'][:,np.newaxis])) 
            self.barycentric[key] = coords.T
            
    def coords_from_basis(self,basis):
        ret = dict()
        for key in self.barycentric.keys():
            coords = np.dot(basis['A'],(self.barycentric[key]).T)+basis['p'][:,np.newaxis]
            ret[key] = coords
        return(ret)
        
class ModelView(object):
    def __init__(self,model):
        import copy
        self.model = model
        self.plot_basis = copy.copy(model.basis)
        #self.plot_basis['p'] = default_rframe_data['p']
        #self.plot_basis['a1'] = default_rframe_data['a1']
        #self.plot_basis['a2'] = default_rframe_data['a2']
        
    def plot(self,basis,plotobject):
        lines = self.model.coords_from_basis(basis)
        self.curves = list()
        for line in lines.values():
            self.curves.append(plotobject.plot(line[0,:],line[1,:]))

    def update_basis(self,basis):
        lines = self.model.coords_from_basis(basis)
        for curve,line in zip(self.curves,lines.values()):
            curve.setData(line[0,:],line[1,:])

    def basis_changed(self,roi):
        pnts = roi.saveState()['points']
        p = np.array(pnts[1])

        a1 = np.array(pnts[0])-p
        a2 = np.array(pnts[2])-p

        self.plot_basis['p'] = p
        self.plot_basis['a1'] = a1
        self.plot_basis['a2'] = a2
        self.update_basis(self.plot_basis)

class BasisROI(pg.ROI):
    
    def __init__(self, basis, closed=False, pos=None, **args):
        
        pos = [0,0]
        
        self.closed = closed
        self.segments = []
        pg.ROI.__init__(self, pos, **args)
        
        self.addFreeHandle((basis['p'][0]+basis['a1'][0],basis['p'][1]+basis['a1'][1]))
        self.addFreeHandle((basis['p'][0],basis['p'][1]))
        self.addFreeHandle((basis['p'][0]+basis['a2'][0],basis['p'][1]+basis['a2'][1]))

        for i in range(0, len(self.handles)-1):
            self.addSegment(self.handles[i]['item'], self.handles[i+1]['item'])
            
    def addSegment(self, h1, h2, index=None):
        seg = pg.LineSegmentROI(handles=(h1, h2), pen=self.pen, parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        #seg.sigClicked.connect(self.segmentClicked)
        #seg.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        seg.setZValue(self.zValue()+1)
        for h in seg.handles:
            h['item'].setDeletable(False)
        
    def saveState(self):
        state = pg.ROI.saveState(self)
        state['closed'] = self.closed
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self,state):
        pg.ROI.setState(self,state,update = False)
        #state = pg.ROI.saveState(self)
        for h,p in zip(self.getHandles(),state['points']):
            self.movePoint(h,p)

        self.stateChanged(finish=True)
        return state

class MainWindow(TemplateBaseClass):  

    def __init__(self):
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('muscle imaging browser')
        
        # Create the main window
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.loadfileTree()
        
        #frame view
        self.plt = pg.PlotItem()
        self.ui.frameView.setCentralItem(self.plt)
        self.frameView = pg.ImageItem()
        self.plt.addItem(self.frameView)

        #transform image
        self.transformPlt = pg.PlotItem()
        self.ui.transformImage.setCentralItem(self.transformPlt)
        self.transformImage = pg.ImageItem()
        self.transformPlt.addItem(self.transformImage)

        #gama plot########
        self.gammaPlt = pg.PlotItem()
        self.ui.gammaPlot.setCentralItem(self.gammaPlt)
        self.ui.gammaSlider.valueChanged.connect(self.gammaChange)
        #default gama
        self.gammaf = lambda x: x**1
        self.gammax = np.linspace(0,2,100)
        self.gammaCurve = self.gammaPlt.plot(self.gammax,self.gammaf(self.gammax))
        #self.transformPlot.addItem(self.transformImage)

        #timeSeries
        self.timeSeriesPlt = pg.PlotItem()
        self.ui.timeSeriesPlt.setCentralItem(self.timeSeriesPlt)
        self.tserTrace = self.timeSeriesPlt.plot(np.ones(1000))
        self.tpointLine = pg.InfiniteLine(pos = 0,movable = True)
        self.tpointLine.sigPositionChanged.connect(self.tpointLineMoved)
        self.timeSeriesPlt.addItem(self.tpointLine)

        #load frames button
        self.ui.loadFrames.clicked.connect(self.loadFrames)

        #save data button
        self.ui.saveFit.clicked.connect(self.saveFit)
        self.ui.loadFit.clicked.connect(self.loadFit)

        ##scroll bar
        self.ui.frameScrollBar.valueChanged.connect(self.frameScrollBar_valueChanged)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.frameView)
        self.ui.frameHist.setCentralItem(self.hist)

        #load data
        self.loadData()
        self.current_frame = 0
        self.show()
        
        #self.ui.commentBox
        self.ui.frameNumber.setText(str(self.current_frame))
        self.ui.frameNumber.textEdited.connect(self.frameInput)

        #addEpoch
        self.epochPlots = dict()
        self.epoch_dict = dict()
        self.ui.newEpoch.clicked.connect(self.newEpoch)
        self.ui.saveEpoch.clicked.connect(self.saveEpoch)

        self.ui.epochStart.textEdited.connect(self.updateEpochFromText)
        self.ui.epochEnd.textEdited.connect(self.updateEpochFromText)


    def newEpoch(self):
        name = str(self.ui.epochName.text())
        print name
        if (not(name in self.epoch_dict.keys()) and not(name == '')):
            epoch_range = [self.current_frame,self.current_frame + 100]
            self.epoch_dict[name] = epoch_range
            self.plotEpoch(name)
            ep_plot = self.epochPlots[name]
            sta,stp = ep_plot.getRegion()
            self.ui.epochStart.setText(str(int(sta)))
            self.ui.epochEnd.setText(str(int(stp)))

    def clearEpochs(self):
        for k in self.epoch_dict.keys():
            self.timeSeriesPlt.removeItem(self.epochPlots[k])
            self.epochPlots.pop(k)
            self.epoch_dict.pop(k)

    def plotEpoch(self,k):
        ep = pg.LinearRegionItem(values= self.epoch_dict[k])
        ep.epoch_name = k
        ep.sigRegionChanged.connect(self.updateEpochPlot)
        self.epochPlots[k] = ep
        self.timeSeriesPlt.addItem(ep)
        self.tpointLine.setZValue(ep.zValue()+1)

    def updateEpochPlot(self,ep):
        self.ui.epochName.setText(ep.epoch_name)
        self.updateCurrentEpochState()

    def updateEpochFromText(self):
        k = str(self.ui.epochName.text())
        ep_plot = self.epochPlots[k]
        sta = int(self.ui.epochStart.text())
        stp = int(self.ui.epochEnd.text())
        ep_plot.setRegion((sta,stp))
        self.epoch_dict[k] = [sta,stp]

    def updateCurrentEpochState(self):
        k = str(self.ui.epochName.text())
        ep = self.epoch_dict[k]
        ep_plot = self.epochPlots[k]
        sta,stp = ep_plot.getRegion()
        self.ui.epochStart.setText(str(int(sta)))
        self.ui.epochEnd.setText(str(int(stp)))
        self.epoch_dict[k] = [int(sta),int(stp)]

    def saveEpoch(self):
        flydir = '%s%s/'%(dba.root_dir,self.current_fly)
        f = open(flydir + 'epoch_data.cpkl','wb')
        import cPickle
        cPickle.dump(self.epoch_dict,f)
        print self.epoch_dict

    def frameInput(self,value):
        self.current_frame = int(value)
        self.showFrame()

    def tpointLineMoved(self):
        self.current_frame = int(self.tpointLine.value())
        self.showFrame()

    def gammaChange(self,value):
        gamma = value/50.0
        self.gammaf = lambda x: x**gamma
        #print gamma
        self.gammaCurve.setData(self.gammax,self.gammaf(self.gammax))
        self.showFrame()

    def loadfileTree(self):
        self.ui.fileTree.setColumnCount(1)
        items = []
        #for key,fly in zip(fly_db.keys(),fly_db.values()):
        for key,fly in sorted(fly_db.items()):#zip(fly_db.keys(),fly_db.values()):
            #print key
            try:
                exp1 = fly['experiments'].values()[0]
                exptype = fly['experiments'].keys()[0]
                if 'tiff_data' in exp1.keys():
                    #item_list.append('fly%s'%key)
                    item = QtGui.QTreeWidgetItem(None,['Fly%04d'%int(key)])
                    for img_key in ['images','refstack']:
                        if img_key in exp1['tiff_data'].keys():
                            #data_ref = exp1['tiff_data'][img_key]
                            child = QtGui.QTreeWidgetItem(None,[img_key])
                            child.setData(0,QtCore.Qt.UserRole,key)
                            item.insertChild(0,child)
                            items.append(item)
                            #print (img_key,np.shape(exp1['tiff_data'][img_key]))
                        else:
                            pass
                else:
                    print exp1.keys()
            except KeyError:
                pass
        self.ui.fileTree.insertTopLevelItems(0,items)

    def loadData(self):
        import cPickle
        f = open('model_data.cpkl','rb')
        ###f = open('/media/flyranch/ICRA_2015/model_data.cpkl','rb')
        model_data = cPickle.load(f)
        f.close()

        ########################
        #model_keys = []
        e1 = model_data['e1']
        e2 = model_data['e2']

        muscle_dict = dict()
        for key in model_data.keys():
            if not(key in ['e1','e2']):
                muscle_dict[key] = model_data[key]
        basis = Basis()
        basis['a2'] = e1[1]-e2[0]
        basis['a1'] = e2[1]-e2[0]
        basis['p'] = e2[0]
        thorax = GeometricModel(muscle_dict,basis)
        self.thorax_view = ModelView(thorax)
        self.roi = BasisROI(thorax.basis)
        self.roi.sigRegionChanged.connect(self.thorax_view.basis_changed)
        self.roi.sigRegionChanged.connect(self.affineWarp)

        self.plt.disableAutoRange('xy')
        
        state = self.roi.getState()
        rf = default_rframe_data
        pnts = [(rf['p'][0]+rf['a1'][0],rf['p'][1]+rf['a1'][1]),
                 (rf['p'][0],rf['p'][1]),
                 (rf['p'][0]+rf['a2'][0],rf['p'][1]+rf['a2'][1])]
        state['points'] = pnts
        self.roi.setState(state)
        self.roi.stateChanged()
        self.plt.addItem(self.roi)

        self.thorax_view.plot(self.thorax_view.plot_basis,self.plt)


    def loadFrames(self):
        selection = self.ui.fileTree.selectedItems()[0]
        self.current_fly = selection.parent().text(0)
        fnum = int(self.current_fly.split('Fly')[1])
        print fnum
        #fnum = selection.data(0,QtCore.Qt.UserRole)
        #print 'here'
        #print int(fnum)
        self.images = np.array(fly_db[fnum]['experiments'].values()[0]['tiff_data']['images'])
        ### tfile = tifffile.TiffFile('/media/flyranch/ICRA_2015/110215_fly1_2_MMStack_Pos0.ome.tif')
        ### self.images = tfile.asarray()

        self.maximg = np.max(self.images,axis = 0)
        self.transform_img = self.affineWarp(self.maximg)
        #self.current_fly = selection.parent().text(0)
        print self.current_fly
        flydir = '%s%s/'%(dba.root_dir,self.current_fly)

        tser_data = np.array(fly_db[fnum]['experiments'].values()[0]['tiff_data']['axon_framebase']['wb_frequency'])
        self.tserTrace.setData(tser_data)
        

        try:
            f = open(flydir+'basis_fits.cpkl','rb')
            import cPickle
            basis = cPickle.load(f)
            state = self.roi.getState()
            pnts = [(basis['p'][0]+basis['a1'][0],basis['p'][1]+basis['a1'][1]),
                    (basis['p'][0],basis['p'][1]),
                    (basis['p'][0]+basis['a2'][0],basis['p'][1]+basis['a2'][1])]
            state['points'] = pnts
            self.roi.setState(state)
            self.roi.stateChanged()
            self.ui.commentBox.setPlainText(basis['commentBox'])

        except IOError:
            print 'no file'
            self.ui.commentBox.setPlainText('')

        self.clearEpochs()

        try:
            f = open(flydir + 'epoch_data.cpkl','rb')
            import cPickle
            self.epoch_dict = cPickle.load(f)
            for k in self.epoch_dict.keys():
                self.plotEpoch(k)
            self.ui.epochName.setText(self.epoch_dict.keys()[0])
            self.updateCurrentEpochState()
        except IOError:
            print 'no epoch file'
            self.ui.epochName.setText('')
            self.ui.epochStart.setText('')
            self.ui.epochEnd.setText('')

        #self.frameView.setImage(self.images[0,:,:])
        self.current_frame = 0
        self.showFrame()
        self.transformImage.setImage(self.transform_img.astype(np.float32))
        self.ui.frameScrollBar.setMaximum(np.shape(self.images)[0])
        self.plt.autoRange()
        #set transformImage



    def showFrame(self):
        img = self.gammaf(self.images[self.current_frame,:,:].astype(np.float32))
        self.frameView.setImage(img.astype(np.float32))
        self.ui.frameNumber.setText(str(self.current_frame))
        self.ui.frameScrollBar.setValue(self.current_frame)
        self.tpointLine.setValue(self.current_frame)

    def affineWarp(self,roi):
        src_f = self.thorax_view.plot_basis
        dst_f = self.thorax_view.model.basis

        dst_p0 = dst_f['a1'] + dst_f['p']
        dst_p1 = dst_f['p']
        dst_p2 = dst_f['a2'] + dst_f['p']

        src_p0 = src_f['a1'] + src_f['p']
        src_p1 = src_f['p']
        src_p2 = src_f['a2'] + src_f['p']
        import cv2
        A = cv2.getAffineTransform(np.float32([src_p0,src_p1,src_p2]),np.float32([dst_p0,dst_p1,dst_p2]))
        output_shape = (1024, 1024)
        self.transform_img = cv2.warpAffine(self.maximg.T,A,output_shape).T[:,::-1].astype(np.float32)

        display_img = np.dstack((self.transform_img ,self.transform_img ,self.transform_img ))
        display_img += overlay*0.2
        self.transformImage.setImage(display_img)

    def frameScrollBar_valueChanged(self,value):
        #self.frameView.setImage(self.images[value,:,:])
        self.current_frame = value
        self.showFrame()
        
    def saveFit(self):
        import cPickle
        savedata = dict(self.thorax_view.plot_basis)
        comment_text = self.ui.commentBox.toPlainText()
        savedata['commentBox'] = comment_text

        flydir = '%s%s/'%(dba.root_dir,self.current_fly)
        f = open(flydir+'basis_fits.cpkl','wb')
        cPickle.dump(savedata,f)

        f.close()

    def loadFit(self):
        pass
        #print self.ui.fileTree.selectedItems()[0].data(0,QtCore.Qt.UserRole).toPyObject()
        
win = MainWindow()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    fly_db.close()
