#!/usr/bin/env python

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui#QStringList,QString
import rospy
import rosparam
import rospkg

from muscle_imager.msg import MsgExtractedSignal
from phidgets_daq.msg import phidgetsDAQinterpreted
from Kinefly.msg import MsgFlystate
from std_msgs.msg import Header, String

import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError

sizeImage = 128+1024*1024 # Size of header + data.
qt_tick_freq = 5
app = pg.mkQApp()

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'strip_chart.ui')
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

class MainWindow(TemplateBaseClass):  

    def __init__(self):
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('strip chart browser')
        # Create the main window
        self.ui = WindowTemplate()
        #initialize the items created in designer
        self.ui.setupUi(self)

        #do some ros introspection
        rp = rospkg.RosPack()
        self.package_path = rp.get_path('muscle_imager')
        self.model_path = os.path.join(self.package_path,'models')

        self.buffer_duration_sec = 30.0 #sec
        #Set up subscribers and buffers for the 
        #signals coming in from the unmixer
        self.muscle_plots = dict()
        self.muscle_curves = dict()
        self.muscle_buffers = dict()
        self.muscle_subcribers = dict()
        self.muscle_update_period_sec = 30.0/1000#30ms
        self.muscle_buffer_samples = int(self.buffer_duration_sec/self.muscle_update_period_sec)
        for muscle in ['b1','b2','b3','i1','i2','iii1','iii3','iii24','hg1','hg2','hg3','hg4']:
            self.muscle_plots[muscle] = pg.PlotItem()
            for side in ['left','right']:
                self.muscle_buffers[(side,muscle)] = [np.arange(self.muscle_buffer_samples,dtype = float),
                                           np.ones(self.muscle_buffer_samples,dtype = float)]
                self.ui.__dict__[muscle].setCentralItem(self.muscle_plots[muscle])
                c = {'left':'w','right':'r'}[side]
                self.muscle_curves[(side,muscle)] = self.muscle_plots[muscle].plot(self.muscle_buffers[(side,muscle)][0],
                                                                        self.muscle_buffers[(side,muscle)][1],
                                                                        pen = c)
                if side == 'left':
                    self.muscle_subcribers[(side,muscle)] = rospy.Subscriber('/unmixer_%s/%s'%(side,muscle),
                                            MsgExtractedSignal,
                                            self.muscle_signal_callback_left,   
                                            queue_size=None, 
                                            buff_size=2*sizeImage, 
                                            tcp_nodelay=True)
                elif side == 'right':
                    self.muscle_subcribers[(side,muscle)] = rospy.Subscriber('/unmixer_%s/%s'%(side,muscle),
                                            MsgExtractedSignal,
                                            self.muscle_signal_callback_right,   
                                            queue_size=None, 
                                            buff_size=2*sizeImage, 
                                            tcp_nodelay=True)

        #Signals coming in from the daq
        self.daq_buffers = dict()
        self.daq_plots = dict()
        self.daq_curves = dict()
        self.daq_subscribers = dict()
        self.daq_update_period_sec = rospy.get_param('/phidgets_daq/update_rate_ms')/1000.0
        self.daq_buffer_samples = int(self.buffer_duration_sec/self.daq_update_period_sec)
        self.daq_buffers['freq'] = [np.arange(self.daq_buffer_samples,dtype = float),
                                    np.ones(self.daq_buffer_samples,dtype = float)]
        self.daq_plots['freq'] = pg.PlotItem()
        self.ui.freq.setCentralItem(self.daq_plots['freq'])
        self.daq_curves['freq'] = self.daq_plots['freq'].plot(self.daq_buffers['freq'][0],
                                                              self.daq_buffers['freq'][1])
        self.daq_subscribers['freq'] = rospy.Subscriber('/phidgets_daq/freq',
                                            phidgetsDAQinterpreted,
                                            self.daq_signal_callback,   
                                            queue_size=None, 
                                            buff_size=1000, 
                                            tcp_nodelay=True)


        #Signals produced by kinefly
        self.kinefly_buffers = dict()
        self.kinefly_plots = dict()
        self.kinefly_subscribers = dict()
        self.kinefly_curves = dict()
        self.kfly_update_period_sec = 15.0/1000.0
        self.kfly_buffer_samples = int(self.buffer_duration_sec/self.kfly_update_period_sec)
        self.kinefly_buffers['lmr'] = [np.arange(self.kfly_buffer_samples,dtype = float),
                                       np.ones(self.kfly_buffer_samples,dtype = float)]
        self.kinefly_plots['lmr'] = pg.PlotItem()
        self.kinefly_curves['lmr'] = self.kinefly_plots['lmr'].plot(self.kinefly_buffers['lmr'][0],
                                                              self.kinefly_buffers['lmr'][1])
        self.ui.lmr.setCentralItem(self.kinefly_plots['lmr'])
        self.kinefly_subscribers['flystate'] = rospy.Subscriber('/kinefly/flystate',
                                            MsgFlystate,
                                            self.kinefly_signal_callback,   
                                            queue_size=None, 
                                            buff_size=1000, 
                                            tcp_nodelay=True)
        self.lock = False
        rospy.init_node('strip_chart')

        #update the gui with a Qt timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.qt_tick)
        self.timer.start(qt_tick_freq)
        self.show()

    def daq_signal_callback(self,msg):
        if not(self.lock):
            self.daq_buffers['freq'][0] = np.roll(self.daq_buffers['freq'][0],-1)
            self.daq_buffers['freq'][1] = np.roll(self.daq_buffers['freq'][1],-1)
            self.daq_buffers['freq'][0][-1] = msg.time
            self.daq_buffers['freq'][1][-1] = msg.value

    def kinefly_signal_callback(self,msg):
        if not(self.lock):
            self.kinefly_buffers['lmr'][0] = np.roll(self.kinefly_buffers['lmr'][0],-1)
            self.kinefly_buffers['lmr'][1] = np.roll(self.kinefly_buffers['lmr'][1],-1)
            self.kinefly_buffers['lmr'][0][-1] = msg.header.stamp.to_sec()
            if ((len(msg.left.angles) >0) and (len(msg.right.angles) >0)):
                self.kinefly_buffers['lmr'][1][-1] = msg.left.angles[0] - msg.right.angles[0]
            else:
                self.kinefly_buffers['lmr'][1][-1] = np.nan
                
    def muscle_signal_callback_left(self,msg):
        """recive a MsgExtractedSignal message from the left unmixer, msg"""
        if not(self.lock):
            self.muscle_buffers[('left',msg.muscle)][0] = np.roll(self.muscle_buffers[('left',msg.muscle)][0],-1)
            self.muscle_buffers[('left',msg.muscle)][1] = np.roll(self.muscle_buffers[('left',msg.muscle)][1],-1)
            self.muscle_buffers[('left',msg.muscle)][0][-1] = msg.header.stamp.to_sec()
            self.muscle_buffers[('left',msg.muscle)][1][-1] = msg.value

    def muscle_signal_callback_right(self,msg):
        """recive a MsgExtractedSignal message from the right unmixer, msg"""
        if not(self.lock):
            self.muscle_buffers[('right',msg.muscle)][0] = np.roll(self.muscle_buffers[('right',msg.muscle)][0],-1)
            self.muscle_buffers[('right',msg.muscle)][1] = np.roll(self.muscle_buffers[('right',msg.muscle)][1],-1)
            self.muscle_buffers[('right',msg.muscle)][0][-1] = msg.header.stamp.to_sec()
            self.muscle_buffers[('right',msg.muscle)][1][-1] = msg.value

    def qt_tick(self):
        self.lock = True
        """handle a qt timer tick"""
        for side,muscle in self.muscle_curves.keys():
            self.muscle_curves[(side,muscle)].setData(self.muscle_buffers[(side,muscle)][0]-\
                                                self.muscle_buffers[(side,muscle)][0][0],
                                               self.muscle_buffers[(side,muscle)][1])
        self.daq_curves['freq'].setData((self.daq_buffers['freq'][0]-
                                        self.daq_buffers['freq'][0][0]),
                                        self.daq_buffers['freq'][1])
        self.kinefly_curves['lmr'].setData((self.kinefly_buffers['lmr'][0]-
                                        self.kinefly_buffers['lmr'][0][0]),
                                        self.kinefly_buffers['lmr'][1])
        self.lock = False
        app.processEvents()

win = MainWindow()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    #fly_db.close()
