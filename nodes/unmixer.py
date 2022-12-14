#!/usr/bin/env python

import rospy
import rosparam
import rospkg

from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
#import sys
#print sys.path

import muscle_model as mm
#from muscle_imager import muscle_model as mm
from muscle_imager.msg import Msg2DAffineFrame
from muscle_imager.msg import MsgArrayNumpyND
from muscle_imager.msg import MsgExtractedSignal

from muscle_imager.srv import SrvRefFrame
from muscle_imager.srv import SrvRefFrameResponse

from scipy.optimize import nnls


from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import h5py
import os
import cv2
import time

from scipy.linalg.blas import sgemm, sgemv

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

class Unmixer(object):

    def __init__(self):
        # initialize the node
        rospy.init_node('Unmixer')
        self.namespace = rospy.get_namespace()
        self.nodename = rospy.get_name().rstrip('/')
        # The ModelViewFrame publishes the affine reference frame to use 
        # to transform the muscle model
        self.topicMV = self.nodename + '%s/ModelViewFrame' % self.namespace.rstrip('/')
        rospy.Subscriber(self.topicMV, Msg2DAffineFrame, self.new_frame_callback)
        
        rp = rospkg.RosPack()
        self.package_path = rp.get_path('muscle_imager')
        self.model_path = os.path.join(self.package_path,'models')
        
        i = 0 #just choose the first model in the model path.. for now.
        self.cur_model = os.listdir(self.model_path)[i]
        self.outline_file_name = self.model_path + '/%s/outlines.cpkl'%(self.cur_model)
        self.components_file_name = self.model_path + '/%s/flatened_model.hdf5'%(self.cur_model)

        # load the outline data
        import cPickle
        with open(self.outline_file_name,'rb') as f:
                outlines = cPickle.load(f)
        e1 = outlines['e1']
        e2 = outlines['e2']

        # create the reference frame
        self.confocal_frame = mm.Frame()
        self.confocal_frame['a2'] = e1[1]-e2[0]
        self.confocal_frame['a1'] = e2[1]-e2[0]
        self.confocal_frame['p'] = e2[0]

        # load the components
        model_data = h5py.File(self.components_file_name,'r')
        self.model_dict = dict()
        [self.model_dict.update({key:np.array(value)}) for key,value in model_data.items()]
        self.model_muscle_names = model_data.keys()
        model_data.close()

        # A CvBridge is needed to parse the ros experiments
        self.cvbridge = CvBridge()

        self.caImage  = rospy.Subscriber(rospy.get_param('~image_topic'), 
                                            Image,  
                                            self.ca_image_callback,   
                                            queue_size=None, 
                                            buff_size=2*sizeImage, 
                                            #tcp_nodelay=True)
                                            tcp_nodelay=False)
        
        self.pubImage = rospy.Publisher(self.nodename+'/image_output', 
                                        Image,  queue_size=2)
        
        self.RefFrameServer = rospy.Service(self.nodename+'/RefFrameServer',
        									 SrvRefFrame,
        									 self.serve_ref_frame,
        									 buff_size = 2*16)

        #publish on serving request for reference frame - this is so data can be
        #logged in bagfile when running a script
        # START testing by SCW,. 12/12/22 
        self.topicLogRefFrame = self.nodename + '%s/LogRefFrame' % self.namespace.rstrip('/')
        # self.topicLogRefFrame = '%s/LogRefFrame' % self.namespace.rstrip('/')
        # END testing by SCW,. 12/12/22 
#        if 'left' in self.nodename:
#            self.topicLogRefFrame = '/live_viewer_left/' + '%s/LogRefFrame' % self.namespace.rstrip('/')
#        elif 'right' in self.nodename:
#            self.topicLogRefFrame = '/live_viewer_right/' + '%s/LogRefFrame' % self.namespace.rstrip('/')
        self.PubRefFrame = rospy.Publisher(self.topicLogRefFrame,
        									 Msg2DAffineFrame, 
        									 queue_size = 1000)

    def serve_ref_frame(self,req):
    	#publish for logging
    	header = Header(stamp=rospy.Time.now())
    	rospy.logwarn(self.user_frame['p'])
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


#modified by francesca on 0622 to avoid errors when publishers get delayed
    def ca_image_callback(self,img):
        """unmix an incoming image img"""
        self.ca_image = self.cvbridge.imgmsg_to_cv2(img, 'passthrough').astype(float)
        #from scipy.optimize import nnls
        im_vect = self.ca_image.astype(np.float32).ravel()
        #fits = np.empty((np.shape(self.model_matrix)[0],np.shape(im_vect)[0]))
        if hasattr(self, 'model_inv'):
            #fits = np.dot(self.model_inv,im_vect.T)
            t_fit = time.time()
            fits = np.linalg.multi_dot([self.model_inv,im_vect.T])
            #fits = np.matmul(self.model_inv,im_vect.T)
            #fits = self.model_inv@im_vect.T
            #fits = sgemv(1.0,self.model_inv,im_vect.T)
            #t_fit = time.time()-t_fit
            #if t_fit>0.001:
            #rospy.logwarn(t_fit)

            header = Header(stamp=img.header.stamp)

            for i,m in enumerate(self.muscles):
            	self.muscle_publishers[m].publish(header = header,value = float(fits[i]),muscle = m)

            #for i,m in enumerate(self.muscles):
            #        try:
            #            self.muscle_publishers[m].publish(header = header,value = float(fits[i]),muscle = m)
            #        except:
            #            rospy.logwarn('muscle activity publishers crapping out')


    def new_frame_callback(self,msg):
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
        Ap = np.dot([[s,0.0,0],[0,s,0],[0,0,1]],A)
        Ap = A
        output_shape = (np.array(self.ca_image.shape)*s).astype(int) #confocal shape * the scale
        #output_shape = (np.array(self.ca_image.shape)).astype(int)
        output_shape = (output_shape[0],output_shape[1]) #make the shape a tuple
        self.warped_model_dict = dict()
        for component in msg.components.split(';'):
            self.warped_model_dict[component] = \
                    cv2.warpAffine(self.model_dict[component],
                                   Ap[:-1,:],output_shape).astype(np.float32).T
        self.muscles = self.warped_model_dict.keys()
        tmp = [self.warped_model_dict[m].ravel() for m in self.muscles]
        tmp.append(np.ones_like(self.warped_model_dict[m].ravel()))#background term
        self.muscles.append('bkg')
        self.model_mtrx = np.vstack(tmp)


        self.topicMV = '%s/ModelViewFrame' % self.namespace.rstrip('/')
        self.muscle_publishers = dict()
        for muscle in self.muscles:
            self.muscle_publishers[muscle] = rospy.Publisher(self.nodename+'/' + muscle, 
                                                MsgExtractedSignal,  queue_size=2)
        from numpy.linalg import pinv
        self.model_inv = pinv(self.model_mtrx.T)


if __name__ == '__main__':
    unmx = Unmixer()
    rospy.spin()
