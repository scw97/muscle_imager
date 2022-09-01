#!/usr/bin/env python

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui#QStringList,QString
import rospy
import rosparam
import rospkg
from sensor_msgs.msg import Image
from muscle_imager.msg import Msg2DAffineFrame
from muscle_imager.msg import MsgArrayNumpyND
from muscle_imager.msg import MsgExtractedSignal

from std_msgs.msg import Header, String

import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError




sizeImage = 128+1024*1024 # Size of header + data.

pg.mkQApp()

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'live_viewer.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

#from muscle_imager import muscle_model as mm
import muscle_model as mm

default_rframe_data = {'a1': np.array([ 51.5848967 ,  -5.93928407]),
                       'a2': np.array([ -0.09151179,  88.42505672]),
                       'p': np.array([ 26.66908747,  34.43488385])}

def toNumpyND(np_ndarray):
    msg = MsgArrayNumpyND()
    msg.shape = np.array(np_ndarray.shape).astype(int)
    msg.data = np.ravel(np_ndarray).astype(np.float64)
    return msg

class ModelView(object):

    def __init__(self,model):
        import copy
        self.model = model
        self.plot_frame = copy.copy(model.frame)
        self.curves = None
        self.element_list = []

        self.namespace = rospy.get_namespace()
        self.topicMV = '%s/ModelViewFrame' % self.namespace.rstrip('/')
        self.pubMV = rospy.Publisher(self.topicMV, Msg2DAffineFrame,queue_size = 1000)
        
    def plot(self,frame,plotobject):
        if self.curves:
            for pitem in self.curves:
                plotobject.removeItem(pitem)
        lines = self.model.coords_from_frame(frame)
        self.curves = list()
        for element_name, line in lines.items():
            if element_name in self.element_list:
                self.curves.append(plotobject.plot(line[0,:],line[1,:]))
        
    def publish_ros(self):
        header = Header(stamp=rospy.Time.now())
        self.pubMV.publish(header = header,
                            a1 = toNumpyND(self.plot_frame['a1']),
                            a2 = toNumpyND(self.plot_frame['a2']),
                            A = toNumpyND(self.plot_frame['A']),
                            A_inv = toNumpyND(self.plot_frame['A_inv']),
                            p = toNumpyND(self.plot_frame['p']),
                            components = ';'.join(self.element_list))

    def update_frame(self,frame):
        lines = self.model.coords_from_frame(frame)
        lines = [l for k,l in lines.items() if k in self.element_list]
        if self.curves:
            for curve,line in zip(self.curves,lines):#lines.values()):
                curve.setData(line[0,:],line[1,:])

    def frame_changed(self,roi):
        pnts = roi.saveState()['points']
        p = np.array(pnts[1])
        a1 = np.array(pnts[0])-p
        a2 = np.array(pnts[2])-p

        self.plot_frame['p'] = p
        self.plot_frame['a1'] = a1
        self.plot_frame['a2'] = a2
        print self.plot_frame['A']
        self.update_frame(self.plot_frame)
        self.publish_ros()

class RefrenceFrameROI(pg.ROI):
    
    def __init__(self, frame, closed=False, pos=None, **args):
        pos = [0,0]
        self.closed = closed
        self.segments = []
        pg.ROI.__init__(self, pos, **args)
        
        self.addFreeHandle((frame['p'][0]+frame['a1'][0],frame['p'][1]+frame['a1'][1]))
        self.addFreeHandle((frame['p'][0],frame['p'][1]))
        self.addFreeHandle((frame['p'][0]+frame['a2'][0],frame['p'][1]+frame['a2'][1]))

        for i in range(0, len(self.handles)-1):
            self.addSegment(self.handles[i]['item'], self.handles[i+1]['item'])
            
    def addSegment(self, h1, h2, index=None):
        seg = pg.LineSegmentROI(handles=(h1, h2), 
                                pen=self.pen, 
                                parent=self, 
                                movable=False)
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
        #initialize the items created in designer
        self.ui.setupUi(self)
        
        #frame view
        self.plt = pg.PlotItem()
        self.ui.frameView.setCentralItem(self.plt)
        self.frameView = pg.ImageItem()
        self.plt.addItem(self.frameView)

        #gama plot
        self.gammaPlt = pg.PlotItem()
        self.ui.gammaPlot.setCentralItem(self.gammaPlt)
        self.ui.gammaSlider.valueChanged.connect(self.gammaChange)
        
        #default gama
        self.gammaf = lambda x: x**1
        self.gammax = np.linspace(0,2,100)
        self.gammaCurve = self.gammaPlt.plot(self.gammax,self.gammaf(self.gammax))

        #timeSeries

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.frameView)
        self.ui.frameHist.setCentralItem(self.hist)

        self.componentsModel = QtGui.QStandardItemModel(self.ui.componentsView)
        self.ui.componentsView.setModel(self.componentsModel)
        
        rospy.init_node('muscle_viewer')

 
        rp = rospkg.RosPack()
        self.package_path = rp.get_path('muscle_imager')
        self.model_path = os.path.join(self.package_path,'models')

        self.nodename = rospy.get_name().rstrip('/')
        self.img1 = None
        self.cvbridge = CvBridge()
        #self.pubImage       = rospy.Publisher(self.nodename+'/image_output', Image,  queue_size=2)
        self.subImage = rospy.Subscriber(rospy.get_param('~image_topic'), 
                                            Image,  
                                            self.image_callback,   
                                            queue_size=None, 
                                            buff_size=2*sizeImage, 
                                            tcp_nodelay=True)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.qt_tick)
        self.timer.start(30)
        self.rosimg = None
        self.componentsModel.itemChanged.connect(self.componentsChanged)

        #modelSelector
        self.loadedComponents = list()
        self.updateModelList()
        self.ui.modelselectBox.currentIndexChanged.connect(self.modelSelected)
        self.modelSelected(0)

        #profileSelector
        self.updateProfileList()
        self.ui.profileselectBox.currentIndexChanged.connect(self.profileSelected)
        self.profileSelected(0)
        self.ui.saveProfile.clicked.connect(self.saveProfile)

        #load outlines
        self.loadLines()
        self.show()

    def gammaChange(self,value):
        gamma = value/50.0
        self.gammaf = lambda x: x**gamma
        #print gamma
        self.gammaCurve.setData(self.gammax,self.gammaf(self.gammax))
        #self.showFrame()

    def muscle_signal_callback(self,msg):
        self.muscle_buffers[msg.muscle][0] = np.roll(self.muscle_buffers[msg.muscle][0],-1)
        self.muscle_buffers[msg.muscle][1] = np.roll(self.muscle_buffers[msg.muscle][1],-1)
        self.muscle_buffers[msg.muscle][0][-1] = msg.header.stamp.to_sec()
        self.muscle_buffers[msg.muscle][1][-1] = msg.value
        
    def updateModelList(self):
        import os
        for mstr in os.listdir(self.model_path):
            self.ui.modelselectBox.addItem(mstr)

    def image_callback(self,rosimg):
        self.rosimg = rosimg
        #self.show_img = self.cvbridge.imgmsg_to_cv2(rosimg, 'passthrough').astype(float)
        
    def qt_tick(self):
        if not(self.rosimg is None):
            img = self.cvbridge.imgmsg_to_cv2(self.rosimg, 'passthrough').astype(float)
            img = self.gammaf(img)
            img = np.fliplr(np.transpose(img)) # change made by Johan
            self.frameView.setImage(img)
        else:
            pass
        #for muscle in self.muscle_curves.keys():
        #    self.muscle_curves[muscle].setData(self.muscle_buffers[muscle][0]-\
        #                                       self.muscle_buffers[muscle][0][0],
        #                                       self.muscle_buffers[muscle][1])
        #try:
        #self.thorax_view.publish_ros()

    def profileSelected(self,i):
        import cPickle
        profile = self.ui.profileselectBox.currentText()
        with open(self.model_path + '/%s/profiles/%s'%(self.cur_model,profile),'rb') as f:
            profile_data = cPickle.load(f)
        for component in self.loadedComponents:
            if component['name'] in profile_data['selected_components']:
                component['checkbox'].setCheckState(True)
            else:
                component['checkbox'].setCheckState(False)
        self.ui.profileName.setText(profile)
        
        try:
            self.thorax_view.publish_ros()
        except AttributeError:
            pass

    def updateProfileList(self):
        import os
        profile_list = os.listdir(self.model_path + '/%s/profiles'%(self.cur_model))
        if len(profile_list) == 0:
            #print 'creating default profile'
            import cPickle
            with open('models/%s/profiles/default.cpkl'%(self.cur_model),'wb') as f:
                cPickle.dump({'selected_components':[]},f)
            self.updateProfileList()
        else:
            for profile in profile_list:
                self.ui.profileselectBox.addItem(profile)
        index = self.ui.profileselectBox.findText('default.cpkl', QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.ui.profileselectBox.setCurrentIndex(index)

    def saveProfile(self):
        profile_dir = self.model_path + '/%s/profiles/'%(self.cur_model)
        name = str(self.ui.profileName.text())
        f = open(profile_dir + name,'wb')
        import cPickle
        cPickle.dump({'selected_components':self.thorax_view.element_list},f)
        #self.updateProfileList()

    def componentsChanged(self):
        # If the changed item is not checked, don't bother checking others
        #if not item.checkState():
        #    return
        # Loop through the items until you get None, which
        # means you've passed the end of the list
        i = 0
        item_list = list()
        while self.componentsModel.item(i):
            if self.componentsModel.item(i).checkState():
                item_list.append(i)
                #return
            i += 1
        #skeys = self.signalshelf.keys()
        self.checked_signals = [self.loadedComponents[i]['name'] for i in item_list]

        self.thorax_view.element_list = self.checked_signals
        self.thorax_view.plot(self.thorax_view.plot_frame,self.plt)
        self.roi.stateChanged()

        
        #self.update_tser_plot()

    def modelSelected(self,i):
        import cPickle
        self.cur_model = os.listdir(self.model_path)[i]
        #print self.cur_modelcomponentsModel
        with open(self.model_path + '/%s/outlines.cpkl'%(self.cur_model),'rb') as f:
            self.outlines = cPickle.load(f)
        for key in self.outlines.keys():
            #print key
            item = QtGui.QStandardItem(key)
            #check = 1 if np.random.randint(0, 1) == 1 else 0
            item.setCheckable(True)
            item.setCheckState(False)
            self.loadedComponents.append({'checkbox':item,'name':key})
            self.componentsModel.appendRow(item)
            #self.color_dict[key] = 'r'

    def loadLines(self):
        import cPickle
        #f = open('model_data.cpkl','rb')
        ###f = open('/media/flyranch/ICRA_2015/model_data.cpkl','rb')
        model_data = self.outlines
        #f.close()

        ########################
        #model_keys = []
        e1 = model_data['e1']
        e2 = model_data['e2']

        muscle_dict = dict()
        for key in model_data.keys():
            if not(key in ['e1','e2']):
                muscle_dict[key] = model_data[key]
        frame = mm.Frame()
        frame['a2'] = e1[1]-e2[0]
        frame['a1'] = e2[1]-e2[0]
        frame['p'] = e2[0]
        thorax = mm.GeometricModel(muscle_dict,frame)
        self.thorax_view = ModelView(thorax)
        self.roi = RefrenceFrameROI(thorax.frame)
        self.roi.sigRegionChanged.connect(self.thorax_view.frame_changed)
        #self.roi.sigRegionChanged.connect(self.affineWarp)

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

        self.thorax_view.plot(self.thorax_view.plot_frame,self.plt)


win = MainWindow()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    #fly_db.close()
