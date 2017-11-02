#!/usr/bin/env python

import dynamic_reconfigure.client
import rospy

rospy.init_node('myconfig_py', anonymous=True)
ca_cam_params_1 = {'trigger_mode':'mode14','enable_trigger':True,}
ca_cam_params_2 = {'trigger_mode':'mode1','enable_trigger':True,'trigger_polarity':0}
ca_cam_params_3 = {'trigger_mode':'mode1','enable_trigger':True}

c1 = dynamic_reconfigure.client.Client('/ca_camera_right/camera_nodelet')
c2 = dynamic_reconfigure.client.Client('/ca_camera_left/camera_nodelet')
c3 = dynamic_reconfigure.client.Client('/kine_camera_1/camera_nodelet')

config = c1.update_configuration(ca_cam_params_2)
config = c2.update_configuration(ca_cam_params_2)
config = c1.update_configuration(ca_cam_params_1)
config = c2.update_configuration(ca_cam_params_1)
config = c1.update_configuration(ca_cam_params_3)
config = c2.update_configuration(ca_cam_params_3)


#config = c3.update_configuration({'trigger_mode':'mode1','enable_trigger':True,'trigger_polarity':1})
config = c3.update_configuration({'trigger_mode':'mode14','enable_trigger':False,'frame_rate':50})
#config = c3.update_configuration({'enable_trigger':False,})
#config = c3.update_configuration({'trigger_mode':'mode1'})