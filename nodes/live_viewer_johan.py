#!/usr/bin/env python

from __future__ import print_function

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidgetItem, QVBoxLayout, QFileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import rospy
import rosparam
import rospkg
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage #ivo
from muscle_imager.msg import Msg2DAffineFrame
from muscle_imager.msg import MsgArrayNumpyND
from muscle_imager.msg import MsgExtractedSignal
from muscle_imager.msg import MsgArrayNumpyUint8

from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Header, String, Bool, UInt32, Float32
from rospy_tutorials.msg import Floats

import numpy as np
import os
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import copy
import h5py
import cPickle
import datetime

from scipy.linalg import blas as FB
from scipy.sparse import csc_matrix
from scipy.signal import correlate
from scipy.signal import savgol_filter
from numpy.linalg import pinv

from live_viewer_joh import Ui_MainWindow

#from muscle_imager import muscle_model as mm
import muscle_model as mm

from Kinefly.msg import MsgFlystate

from muscle_imager.msg import Msg2DAffineFrame
from muscle_imager.msg import MsgArrayNumpyND
from muscle_imager.msg import MsgExtractedSignal
#from muscle_imager.msg import MsgLabeledImg
#from muscle_imager.msg import MsgTriggerActivation
#from muscle_imager.msg import MsgTrigger

from muscle_imager.srv import SrvRefFrame
from muscle_imager.srv import SrvRefFrameResponse

#default_rframe_data = {'a1': np.array([ 51.5848967 ,  -5.93928407]),
#					'a2': np.array([ -0.09151179,  88.42505672]),
#					'p': np.array([ 26.66908747,  34.43488385])}
#default_rframe_data = {'a1': np.array([ 96.60445162,    7.45764452]),
#						'a2': np.array([ 15.58873457, -134.39742286]),
#						'p':  np.array([ 65.02003245,  181.01414001])}

default_rframe_data = {'a1': np.array([115.5, 15.3]),
				       'a2': np.array([-23., -142.]),
					    'p': np.array([67.1, 189.9])}

sizeImage = 128+1024*1024

def toNumpyND(np_ndarray):
	msg = MsgArrayNumpyND()
	msg.shape = np.array(np_ndarray.shape).astype(int)
	msg.data = np.ravel(np_ndarray)
	return msg

def fromNumpyND(msg):
	#msg = MsgArrayNumpyND()
	#msg.shape = np.array(np_ndarray.shape).astype(int)
	#msg.data = np.ravel(np_ndarray)
	return np.reshape(msg.data,msg.shape).astype(np.float64)

def toNumpyUint8(np_ndarray):
	msg = MsgArrayNumpyUint8()
	msg.shape = np.array(np_ndarray.shape).astype(int)
	uint8_array = np_ndarray.astype(np.int16)
	msg.data = np.ravel(uint8_array)
	return msg

def fromNumpyUint8(msg):
	#msg = MsgArrayNumpyND()
	#msg.shape = np.array(np_ndarray.shape).astype(int)
	#msg.data = np.ravel(np_ndarray)
	return np.reshape(msg.data,msg.shape).astype(np.int16)

class ModelView(object):

	def __init__(self,model):
		import copy
		self.model = model
		self.plot_frame = copy.copy(model.frame)
		self.curves = None
		self.element_list = []

		self.namespace = rospy.get_namespace()
		self.nodename = rospy.get_name().rstrip('/')
		self.topicMV = self.nodename+'%s/ModelViewFrame' % self.namespace.rstrip('/')
		self.pubMV = rospy.Publisher(self.topicMV, Msg2DAffineFrame,queue_size = 1000)

		self.u_min = 10000
		self.v_min = 10000
		self.u_max = 0
		self.v_max = 0

		#self.t_last = time.time()
		
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
		self.u_min = 10000
		self.v_min = 10000
		self.u_max = 0
		self.v_max = 0
		if self.curves:
			for curve,line in zip(self.curves,lines):#lines.values()):
				curve.setData(line[0,:],line[1,:])
				u_curve_min = np.amin(line[0,:])
				if u_curve_min < self.u_min:
					self.u_min = int(u_curve_min)
				u_curve_max = np.amax(line[0,:])
				if u_curve_max > self.u_max:
					self.u_max = int(u_curve_max)
				v_curve_min = np.amin(line[1,:])
				if v_curve_min < self.v_min:
					self.v_min = int(v_curve_min)
				v_curve_max = np.amax(line[1,:])
				if v_curve_max > self.v_max:
					self.v_max = int(v_curve_max)

	def frame_changed(self,roi):
		pnts = roi.saveState()['points']
		p = np.array(pnts[1])
		a1 = np.array(pnts[0])-p
		a2 = np.array(pnts[2])-p

		self.plot_frame['p'] = p
		self.plot_frame['a1'] = a1
		self.plot_frame['a2'] = a2
		self.update_frame(self.plot_frame)
		self.publish_ros()

		rospy.logwarn("a1 " + str(a1[0])[0:5] + ", " + str(a1[1])[0:4] + " a2 " + str(a2[0])[0:4] + ", " + str(a2[1])[0:5] + " p " + str(p[0])[0:4] + ", " + str(p[1])[0:5])  #ivo
		#rospy.logwarn("dt frame_changed: " + str(time.time()-self.t_last))
		#self.t_last = time.time()

class ReferenceFrameROI(pg.ROI):
	
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

class FlyamiViewer(QtWidgets.QMainWindow, Ui_MainWindow, QObject):

	def __init__(self, parent=None):
		super(FlyamiViewer,self).__init__(parent)
		self.setupUi(self)

		self.timer_ms = 100
		self.exp_duration = 1800*15000
		self.exp_running = False
		self.exp_mode = 'muscle_triggering'
		self.frame_counter = 0
		self.t_flight = 0
		self.t_flight_start = 0
		self.t_min_closed_loop = 10*15000 # fly has to be flying for 10 seconds before showing an open loop presentation
		self.t_open_loop = 5*15000 # open loop presentation for 5 seconds (30000 frames)
		self.t_start_open_loop = 0
		self.t_pause_open_loop = 60*15000 # Don't show an open loop stimulus for 60 seconds
		#self.N_trigger_total = 16
		self.N_trigger_total = 8
		self.t_start_exp = 0
		self.img_size = [200,200]

		self.m_buffer_len = 2000 # roughly 20 seconds of data
		self.img_buffer_size = 3
		self.img_buffer_ind = 0
		self.image_buffer = np.zeros((self.img_size[0]*self.img_size[1],self.img_buffer_size),dtype=np.float32)

		#----------------------------------------------------------------------------------------
		# Ca image viewer
		#----------------------------------------------------------------------------------------
		
		# ca image view
		self.caPlt = pg.PlotItem()
		self.ca_img_view.setCentralItem(self.caPlt)
		self.ca_View = pg.ImageItem()
		self.caPlt.addItem(self.ca_View)
		rospy.logwarn('caView')

		#gamma plot
		self.gammaPlt = pg.PlotItem()
		self.gamma_view.setCentralItem(self.gammaPlt)
		self.gamma_slider.setMinimum(0)
		self.gamma_slider.setMaximum(100)
		self.gamma_slider.setValue(50)
		self.gamma_slider.valueChanged.connect(self.gammaChange)
		rospy.logwarn('gamma_slider')
		
		#default gamma
		self.gammaf = lambda x: x**1
		self.gammax = np.linspace(0,2,100)
		self.gammaCurve = self.gammaPlt.plot(self.gammax,self.gammaf(self.gammax))
		rospy.logwarn('gamma_curve')

		# Contrast/color control
		self.hist = pg.HistogramLUTItem()
		self.hist.setImageItem(self.ca_View)
		self.hist_view.setCentralItem(self.hist)
		rospy.logwarn('hist view')

		self.componentsModel = QtGui.QStandardItemModel(self.componentsView)
		self.componentsView.setModel(self.componentsModel)
		rospy.logwarn('components view')

		# initialize node
		rospy.init_node('muscles')
		self.namespace = rospy.get_namespace()
		self.nodename = rospy.get_name().rstrip('/')
		rospy.logwarn('init node')
		
		rospy.logwarn('')
		rospy.logwarn('node name:')
		rospy.logwarn(self.nodename)
		rospy.logwarn('')
		
		_translate = QtCore.QCoreApplication.translate
		self.setWindowTitle(_translate("MainWindow", self.nodename))

		rp = rospkg.RosPack()
		self.package_path = rp.get_path('muscle_imager')
		self.model_path = os.path.join(self.package_path,'models')

		# Ca image view:
		self.img1 = None
		self.cvbridge = CvBridge()
		#self.pubImage       = rospy.Publisher(self.nodename+'/image_output', Image,  queue_size=2)
		if 'left' in self.nodename:
			self.subImage = rospy.Subscriber('/ca_camera_left/image_raw', 
				Image,  
				self.image_callback,
				queue_size=None, 
				buff_size=2*sizeImage, 
				tcp_nodelay=True)
		elif 'right' in self.nodename:
			self.subImage = rospy.Subscriber('/ca_camera_right/image_raw', 
				Image,  
				self.image_callback,
				queue_size=None, 
				buff_size=2*sizeImage, 
				tcp_nodelay=True)
		# FPS display:
		self.time_fps = time.time()
		self.fps_counter = 0
		self.fps_period = 500
		self.ca_fps_disp.setText(str(0.0))
		rospy.logwarn('ca view')

		# Qt timer
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.qt_tick)
		self.timer.start(self.timer_ms)
		self.rosimg = None
		rospy.logwarn('qt timer')

		#components changed
		self.componentsModel.itemChanged.connect(self.componentsChanged)
		rospy.logwarn('components changed')

		#modelSelector
		self.loadedComponents = list()
		self.updateModelList()
		self.modelSelectBox.currentIndexChanged.connect(self.modelSelected)
		self.modelSelected(0)
		rospy.logwarn('model selector')

		#profileSelector
		self.updateProfileList()
		self.profileSelectBox.currentIndexChanged.connect(self.profileSelected)
		self.profileSelected(0)
		#self.ui.saveProfile.clicked.connect(self.saveProfile)
		rospy.logwarn('profile selector')

		#load outlines
		self.loadLines()
		self.show()
		rospy.logwarn('load lines')

		#----------------------------------------------------------------------------------------
		# Muscle unmixer
		#----------------------------------------------------------------------------------------

		# The ModelViewFrame publishes the affine reference frame to use 
		# to transform the muscle model
		#self.topicMV = '%s/ModelViewFrame' % self.namespace.rstrip('/')
		self.topicMV = self.nodename + '%s/ModelViewFrame' % self.namespace.rstrip('/')
		#self.topicMV = '/ca_camera/image'
		#print self.topicMV
		rospy.Subscriber(self.topicMV, Msg2DAffineFrame, self.new_frame_callback)
		rospy.logwarn('subscribe msg2daffine')

		rp = rospkg.RosPack()
		self.package_path = rp.get_path('muscle_imager')
		self.model_path = os.path.join(self.package_path,'models')
		rospy.logwarn('package and model paths')
		
		i = 0 #just choose the first model in the model path.. for now.
		self.cur_model = os.listdir(self.model_path)[i]
		self.outline_file_name = self.model_path + '/%s/outlines.cpkl'%(self.cur_model)
		self.components_file_name = self.model_path + '/%s/flatened_model.hdf5'%(self.cur_model)
		rospy.logwarn('components file')

		# load the outline data
		import cPickle
		#rospy.logwarn(self.outline_file_name) #ivo
		with open(self.outline_file_name,'rb') as f:
				outlines = cPickle.load(f)
		e1 = outlines['e1']
		e2 = outlines['e2']
		rospy.logwarn('load outline data')

		# create the reference frame
		self.confocal_frame = mm.Frame()
		self.confocal_frame['a2'] = e1[1]-e2[0]
		self.confocal_frame['a1'] = e2[1]-e2[0]
		self.confocal_frame['p'] = e2[0]
		rospy.logwarn('create reference frame')

		# load the components
		model_data = h5py.File(self.components_file_name,'r')
		self.model_dict = dict()
		[self.model_dict.update({key:np.array(value)}) for key,value in model_data.items()]
		self.model_muscle_names = model_data.keys()
		model_data.close()
		rospy.logwarn('load muscle dictionary')

		# Crop window:
		#self.crop_window = (np.zeros(4)).astype(int)
		#self.crop_window[1] = 2
		#self.crop_window[3] = 2

		# Muscle signal buffer
		self.m_buffer = np.zeros((20,self.m_buffer_len))
		self.m_filtered = np.zeros((20,self.m_buffer_len))
		self.m_buff_ind = 0
		self.m_mean = np.zeros((20,1))
		self.m_std = np.ones((20,1))
		self.hz = 200.0

		# Subscribe to Kinefly flystate
		self.kine_angles = rospy.Subscriber('/kinefly/flystate', MsgFlystate, self.kine_left_callback, queue_size = 1000)
		rospy.logwarn('subscribe /kinefly/flystate')

		# Subscribe to Kinefly image #ivo
		self.kine_images = rospy.Subscriber('/kinefly/image_output', Image, self.kine_image_callback, queue_size = 10)
		rospy.logwarn('subscribe /kinefly/image_output')
		
		# Muscle signal unmixer:
		self.pubImage = rospy.Publisher(self.nodename+'/image_output', Image, queue_size=2)
		rospy.logwarn('publish /image_output')
		
		if 'left' in self.nodename:
			self.caImage  = rospy.Subscriber('/ca_camera_left/image_raw', 
				Image,  
				self.ca_image_callback,   
				queue_size=10, 
				buff_size=10*sizeImage,
				tcp_nodelay=True)
		elif 'right' in self.nodename:
			self.caImage  = rospy.Subscriber('/ca_camera_right/image_raw', 
				Image,  
				self.ca_image_callback,   
				queue_size=10, 
				buff_size=10*sizeImage,
				tcp_nodelay=True)

		self.RefFrameServer = rospy.Service(self.nodename+'/RefFrameServer',
											 SrvRefFrame,
											 self.serve_ref_frame,
											 buff_size = 2*16)
		rospy.logwarn('service RefFrameServer')

		#publish on serving request for reference frame - this is so data can be
		#logged in bagfile when running a scriptself.stop_recording_hdf5_file()
		self.topicLogRefFrame = self.nodename+'%s/LogRefFrame' % self.namespace.rstrip('/')
		self.PubRefFrame = rospy.Publisher(self.topicLogRefFrame,
											 Msg2DAffineFrame, 
											 queue_size = 1000)
		rospy.logwarn('publish RefFrameServer')


		#----------------------------------------------------------------------------------------
		# Muscle signal plot
		#----------------------------------------------------------------------------------------

		self.muscles = []

		# b muscles
		self.b_item = pg.PlotItem()
		self.b_viewer.setCentralItem(self.b_item)

		# i muscles
		self.i_item = pg.PlotItem()
		self.i_viewer.setCentralItem(self.i_item)

		# iii muscles
		self.iii_item = pg.PlotItem()
		self.iii_viewer.setCentralItem(self.iii_item)

		# hg muscles
		self.hg_item = pg.PlotItem()
		self.hg_viewer.setCentralItem(self.hg_item)

		self.update_muscle_pens = True
		self.muscle_t_range = 10.0
		self.muscle_show_range = self.m_buffer_len # only show the last 2000 values of the muscle buffer

		self.muscle_show_step = 1 # skip 4 out of 5 frames while showing muscle buffer

		# Qt timer
		rospy.logwarn('qt timer')
		self.timer2 = QtCore.QTimer()
		self.timer2.timeout.connect(self.update_muscle_plot)
		self.timer2.start(self.timer_ms)
		rospy.logwarn('timer has started')

		self.lock = False
		rospy.on_shutdown(self.OnShutdown_callback)

	def gammaChange(self,value):
		gamma = value/50.0
		self.gammaf = lambda x: x**gamma
		self.gammaCurve.setData(self.gammax,self.gammaf(self.gammax))
		#self.showFrame()

	def updateModelList(self):
		for mstr in os.listdir(self.model_path):
			self.modelSelectBox.addItem(mstr)

	def profileSelected(self,i):
		profile = self.profileSelectBox.currentText()
		with open(self.model_path + '/%s/profiles/%s'%(self.cur_model,profile),'rb') as f:
			profile_data = cPickle.load(f)
		for component in self.loadedComponents:
			if component['name'] in profile_data['selected_components']:
				component['checkbox'].setCheckState(True)
			else:
				component['checkbox'].setCheckState(False)
		self.profileName.setText(profile)
		
		try:
			self.thorax_view.publish_ros()
		except AttributeError:
			pass

	def updateProfileList(self):
		profile_list = os.listdir(self.model_path + '/%s/profiles'%(self.cur_model))
		if len(profile_list) == 0:
			#print 'creating default profile'
			import cPickle
			with open('models/%s/profiles/default.cpkl'%(self.cur_model),'wb') as f:
				cPickle.dump({'selected_components':[]},f)
			self.updateProfileList()
		else:
			for profile in profile_list:
				self.profileSelectBox.addItem(profile)
		index = self.profileSelectBox.findText('default.cpkl', QtCore.Qt.MatchFixedString)
		if index >= 0:
			self.profileSelectBox.setCurrentIndex(index)

	def image_callback(self,rosimg):
		self.rosimg = rosimg

	def kine_left_callback(self,msg):
		angle_left  = msg.left.angles
		angle_right = msg.right.angles
		angle_head  = msg.head.angles #ivo
		angle_aux   = msg.aux.intensity #ivo
		if self.exp_running:
			if angle_left:
				try:
					angle_double = np.double(angle_left[0])
					#rospy.logwarn("angle left: "+str(angle_double))
					#self.h5_file.create_dataset('kine_left_'+str(self.frame_count),data=angle_double)
				except:
					pass
			if angle_right:
				try:
					angle_double = np.double(angle_right[0])
					#rospy.logwarn("angle right: "+str(angle_double))
					#self.h5_file.create_dataset('kine_right_'+str(self.frame_count),data=angle_double)
				except:
					pass
			if angle_head: #ivo
				try:
					angle_double = np.double(angle_head[0])
					#self.h5_file.create_dataset('kine_head_'+str(self.frame_count),data=angle_double)
				except:
					pass
			if angle_aux: #ivo
				try:
					angle_double = np.double(angle_aux)
					#self.h5_file.create_dataset('kine_aux_'+str(self.frame_count),data=angle_double)
				except:
					pass

	def kine_image_callback(self,img):
		if not(img is None):
			kine_image = self.cvbridge.imgmsg_to_cv2(img, 'passthrough')
			if self.exp_running:
				try:
					img_uint8 = np.array(kine_image).astype(np.uint8)
					#self.h5_file.create_dataset('kine_frame_' + str(self.frame_count),data=img_uint8,dtype='uint8')
				except:
					pass
			#rospy.logwarn(img_uint8)

	def qt_tick(self):
		self.lock = True
		if not(self.rosimg is None):
			img = self.cvbridge.imgmsg_to_cv2(self.rosimg, 'passthrough').astype(float)
			#img = self.gammaf(img)
			#img_t = cv2.flip(cv2.transpose(img),flipCode=0)
			img_t = cv2.flip(cv2.flip(cv2.transpose(img),flipCode=0),flipCode=1)
			self.ca_View.setImage(img_t)
		else:
			pass
		self.lock = False

	def componentsChanged(self):
		# If the changed item is not checked, don't btoNumpyUint8(other checking others
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
		self.thorax_view.plot(self.thorax_view.plot_frame,self.caPlt)
		self.roi.stateChanged()
		#self.update_tser_plot()

	def modelSelected(self,i):
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
		self.roi = ReferenceFrameROI(thorax.frame)
		self.roi.sigRegionChanged.connect(self.thorax_view.frame_changed)
		#self.roi.sigRegionChanged.connect(self.affineWarp)

		self.caPlt.disableAutoRange('xy')
		
		state = self.roi.getState()
		rf = default_rframe_data
		pnts = [(rf['p'][0]+rf['a1'][0],rf['p'][1]+rf['a1'][1]),
				 (rf['p'][0],rf['p'][1]),
				 (rf['p'][0]+rf['a2'][0],rf['p'][1]+rf['a2'][1])]
		state['points'] = pnts
		self.roi.setState(state)
		self.roi.stateChanged()
		self.caPlt.addItem(self.roi)

		self.thorax_view.plot(self.thorax_view.plot_frame,self.caPlt)

	def calc_fps(self):
		self.fps_counter += 1
		if (self.fps_counter%self.fps_period==0):
			self.hz = (1.0*self.fps_period)/(time.time()-self.time_fps)
			self.ca_fps_disp.setText(str('%.2f' % self.hz))
			self.time_fps = time.time()
			self.fps_counter = 0

	def serve_ref_frame(self,req):
		#publish for logging
		header = Header(stamp=rospy.Time.now())
		self.PubRefFrame.publish(header = header,
			a1 = toNumpyND(self.user_frame['a1']),
			a2 = toNumpyND(self.user_frame['a2']),
			A = toNumpyND(self.user_frame['A']),
			A_inv = toNumpyND(self.user_frame['A_inv']),
			p = toNumpyND(self.user_frame['p']),
			components = ';'.join(self.muscles))
		return SrvRefFrameResponse(a1 = toNumpyND(self.user_frame['a1']),
			a2 = toNumpyND(self.user_frame['a2']),
			A = toNumpyND(self.user_frame['A']),
			A_inv = toNumpyND(self.user_frame['A_inv']),
			p = toNumpyND(self.user_frame['p']),
			components = ';'.join(self.muscles))

	def ca_image_callback(self,img):
		self.lock = True
		if self.frame_counter%10==0:
			"""unmix an incoming image img"""
			img_cv = self.cvbridge.imgmsg_to_cv2(img, 'passthrough')
			img_t = cv2.flip(cv2.flip(cv2.transpose(img_cv),flipCode=0),flipCode=1)
			self.ca_image = img_t
			header = Header(stamp=img.header.stamp)
			im_vect = self.ca_image.ravel()
			im_vect.astype(np.float32)
			# Compute muscle activities
			try:
				#fits = np.linalg.multi_dot([self.model_inv,im_vect.T])
				fits = np.dot(self.model_inv,im_vect.T)
				#fits = np.matmul(self.model_inv,im_vect.T)
				#fits = np.zeros([1])
			except:
				fits = np.zeros([1])
			try:
				if self.m_buff_ind >= self.m_buffer_len:
					self.m_buff_ind = 0
				self.m_buffer[0,self.m_buff_ind] = time.time()
				for i,m in enumerate(self.muscles):
					self.muscle_publishers[m].publish(header = header, value = float(fits[i]), muscle = m)
					self.m_buffer[i+1,self.m_buff_ind] = fits[i]
				self.m_buff_ind += 1
			except:
				pass
			# moving on
		self.lock = False
		self.calc_fps()
		self.frame_counter += 1

	def new_frame_callback(self,msg):
		if not self.lock:
			"""update the model when the reference frame changes... perform
			the affine warping"""
			self.user_frame = mm.Frame(a1 = fromNumpyND(msg.a1),
								  a2 = fromNumpyND(msg.a2),
								  A = fromNumpyND(msg.A),
								  p = fromNumpyND(msg.p),
								  A_inv = fromNumpyND(msg.A_inv))

			# A is the transform that will be used to transform the 
			# confocal frame into the user frame
			A = self.user_frame.get_transform(self.confocal_frame)

			# Option to compose with a scaling, not used, but here for future flexiblity..
			# for instance if we want to work with smaller images to speed things up
			s = 1.0
			Ap = np.dot([[s,0.0,0.0],[0.0,s,0.0],[0.0,0.0,1.0]],A)
			Ap = A

			output_shape = (np.array(self.ca_image.shape)).astype(int)
			output_shape = (output_shape[0],output_shape[1]) #make the shape a tuple
			self.warped_model_dict = dict()
			for component in msg.components.split(';'):
				self.warped_model_dict[component] = cv2.warpAffine(self.model_dict[component], Ap[:-1,:],output_shape).astype(np.float32).T
			self.muscles = self.warped_model_dict.keys()
			tmp = [self.warped_model_dict[m].ravel() for m in self.muscles]
			tmp.append(np.ones_like(self.warped_model_dict[m].ravel()))#background term
			self.muscles.append('bkg')
			self.model_mtrx = np.vstack(tmp)

			self.topicMV = self.nodename + '%s/ModelViewFrame' % self.namespace.rstrip('/')
			self.muscle_publishers = dict()
			for muscle in self.muscles:
				self.muscle_publishers[muscle] = rospy.Publisher(self.nodename + '/' + muscle, MsgExtractedSignal,  queue_size=2)
			self.model_inv = pinv(self.model_mtrx.T)
			self.model_inv.astype(np.float32)
			self.update_muscle_pens = True
			#rospy.logwarn(self.muscles)

	def update_muscle_plot(self):
		# Set pens:
		t_range = [-self.muscle_show_range,0]
		z_range = [-10.0,10.0]
		#z_range = [-0.4,0.4] #ivo
		if self.muscles:
			if self.update_muscle_pens:
				t_data = np.arange(-self.muscle_show_range,0,self.muscle_show_step)
				self.b_item.clear()
				self.i_item.clear()
				self.iii_item.clear()
				self.hg_item.clear()
				self.b_item.addLegend()
				self.i_item.addLegend()
				self.iii_item.addLegend()
				self.hg_item.addLegend()
				for i, muscle in enumerate(self.muscles):
					#rospy.logwarn("muscle: " + str(muscle))
					#m_data = self.m_buffer[i+1,range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					#m_data = self.m_filtered[i+1,range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					# Muscle data:
					if self.m_buff_ind==0:
						m_data = self.m_buffer[i+1,range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					else:
						m_data_tot = np.roll(self.m_buffer[i+1,:],self.m_buffer_len-self.m_buff_ind-1)
						m_data = m_data_tot[range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					if muscle[0] == 'b':
						if muscle == 'b1':
							self.b1_pen = self.b_item.plot(x=t_data,y=m_data,name='b1')
							self.b1_pen.setPen((255,0,0))
						elif muscle == 'b2':
							self.b2_pen = self.b_item.plot(x=t_data,y=m_data,name='b2')
							self.b2_pen.setPen((0,255,0))
						elif muscle == 'b3':
							self.b3_pen = self.b_item.plot(x=t_data,y=m_data,name='b3')
							self.b3_pen.setPen((0,0,255))
						self.b_item.setLabel('left','z-score')
						self.b_item.setLabel('bottom','wingbeats', units='wb')
						self.b_item.setXRange(t_range[0],t_range[1])
						self.b_item.setYRange(z_range[0],z_range[1])
					elif muscle[0] == 'i' and not muscle[1] == 'i':
						if muscle == 'i1':
							self.i1_pen = self.i_item.plot(x=t_data,y=m_data,name='i1')
							self.i1_pen.setPen((255,0,0))
						elif muscle == 'i2':
							self.i2_pen = self.i_item.plot(x=t_data,y=m_data,name='i2')
							self.i2_pen.setPen((0,0,255))
						self.i_item.setLabel('left','z-score')
						self.i_item.setLabel('bottom','wingbeats', units='wb')
						self.i_item.setXRange(t_range[0],t_range[1])
						self.i_item.setYRange(z_range[0],z_range[1])
					elif muscle[0] == 'i' and muscle[1] == 'i':
						if muscle == 'iii1':
							self.iii1_pen = self.iii_item.plot(x=t_data,y=m_data,name='iii1')
							self.iii1_pen.setPen((255,0,0))
						elif muscle == 'iii24':
							self.iii24_pen = self.iii_item.plot(x=t_data,y=m_data,name='iii24')
							self.iii24_pen.setPen((0,255,0))
						elif muscle == 'iii3':
							self.iii3_pen = self.iii_item.plot(x=t_data,y=m_data,name='iii3')
							self.iii3_pen.setPen((0,0,255))
						self.iii_item.setLabel('left','z-score')
						self.iii_item.setLabel('bottom','wingbeats', units='wb')
						self.iii_item.setXRange(t_range[0],t_range[1])
						self.iii_item.setYRange(z_range[0],z_range[1])
					elif muscle[0] == 'h':
						if muscle == 'hg1':
							self.hg1_pen = self.hg_item.plot(x=t_data,y=m_data,name='hg1')
							self.hg1_pen.setPen((255,0,0))
						elif muscle == 'hg2':
							self.hg2_pen = self.hg_item.plot(x=t_data,y=m_data,name='hg2')
							self.hg2_pen.setPen((0,255,0))
						elif muscle == 'hg3':
							self.hg3_pen = self.hg_item.plot(x=t_data,y=m_data,name='hg3')
							self.hg3_pen.setPen((0,0,255))
						elif muscle == 'hg4':
							self.hg4_pen = self.hg_item.plot(x=t_data,y=m_data,name='hg4')
							self.hg4_pen.setPen((0,255,255))
						self.hg_item.setLabel('left','z-score')
						self.hg_item.setLabel('bottom','wingbeats', units='wb')
						self.hg_item.setXRange(t_range[0],t_range[1])
						self.hg_item.setYRange(z_range[0],z_range[1])
				self.update_muscle_pens = False
			else:
				# Update values:
				t_data = np.arange(-self.muscle_show_range,0,self.muscle_show_step)
				for i, muscle in enumerate(self.muscles):
					# Muscle data:
					if self.m_buff_ind==0:
						m_data = self.m_buffer[i+1,range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					else:
						m_data_tot = np.roll(self.m_buffer[i+1,:],self.m_buffer_len-self.m_buff_ind-1)
						m_data = m_data_tot[range(self.m_buffer_len-self.muscle_show_range,self.m_buffer_len,self.muscle_show_step)]
					# Select muscle line:
					if muscle=='b1':
						self.b1_pen.setData(x=t_data,y=m_data)
					elif muscle=='b2':
						self.b2_pen.setData(x=t_data,y=m_data)
					elif muscle=='b3':
						self.b3_pen.setData(x=t_data,y=m_data)
					elif muscle=='i1':
						self.i1_pen.setData(x=t_data,y=m_data)
					elif muscle=='i2':
						self.i2_pen.setData(x=t_data,y=m_data)
					elif muscle=='iii1':
						self.iii1_pen.setData(x=t_data,y=m_data)
					elif muscle=='iii24':
						self.iii24_pen.setData(x=t_data,y=m_data)
					elif muscle=='iii3':
						self.iii3_pen.setData(x=t_data,y=m_data)
					elif muscle=='hg1':
						self.hg1_pen.setData(x=t_data,y=m_data)
					elif muscle=='hg2':
						self.hg2_pen.setData(x=t_data,y=m_data)
					elif muscle=='hg3':
						self.hg3_pen.setData(x=t_data,y=m_data)
					elif muscle=='hg4':
						self.hg4_pen.setData(x=t_data,y=m_data)
		
	def OnShutdown_callback(self):
		rospy.logwarn('User shutdown')

# ----------------------------------------------------------------------

def appMain():
	app = QtWidgets.QApplication(sys.argv)
	mainWindow = FlyamiViewer()
	mainWindow.show()
	app.exec_()

# ----------------------------------------------------------------------
if __name__ == '__main__':
	appMain()
