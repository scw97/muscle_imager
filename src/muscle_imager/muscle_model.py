#!/usr/bin/env python

##Library for aligning muscle models to the thorax of the fly
import numpy as np

database_path = '/media/analysis-code/flight-muscles/experimental/mn_expression_matrix_plot/line_database.cpkl'

def get_line_database(line_name):
    line_name = line_name.split('_GFP')[0].split('GMR')[1]
    import cPickle
    import os
    f = open(database_path,'rb')
    line_database = cPickle.load(f)
    f.close()
    return line_database

def get_muscle_list(line_name):
    line_database = get_line_database(line_name)
    ln = line_name.split('_GFP')[0].split('GMR')[1]
    muscle_names = list()
    for key in line_database[ln].keys():
        if line_database[ln][key] > 0:
            muscle_names.append(key)
    muscle_names = sorted(muscle_names)
    #muscle_names = sorted(get_muscle_list(line_name))
    return muscle_names

class Frame(dict):    
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

    def load(self,filepath):
        """load from a pickled dictionary"""
        import cPickle
        pkname = filepath
        f = open(pkname);data = cPickle.load(f);f.close()
        for key in data.keys():
            self[key] = data[key]

    def get_transform(self,other):
        """get transform into self from other frame"""
        A1 = np.hstack((self['A'],self['p'][:,np.newaxis]))
        A2 = np.hstack((other['A'],other['p'][:,np.newaxis]))
        A1 = np.vstack((A1,[0,0,1]))
        A2 = np.vstack((A2,[0,0,1]))
        return(np.dot(A1,np.linalg.inv(A2)))

class GeometricModel(object):   
    def __init__(self,lines = None,frame = None,filepath = None):
        if not(filepath is None):
            self.load(filepath)
        else:
            self.construct(lines,frame)

    def construct(self,lines,frame):
        self.lines = lines
        self.frame = frame
        ## put lines in barycentric coords
        self.barycentric = dict()
        for key in self.lines.keys():
            coords = np.dot(self.frame['A_inv'],(self.lines[key]-self.frame['p'][:,np.newaxis])) 
            self.barycentric[key] = coords.T
            
    def load(self,filepath):
        """load from a pickled dictionary"""
        import cPickle as cpkl
        fi = open(filepath)
        model_data = cpkl.load(fi)
        fi.close()
        e1 = model_data['e1']
        e2 = model_data['e2']
        muscles = dict()
        for key in model_data.keys():
            if not(key in ['e1','e2']):
                muscles[key] = model_data[key]
        confocal_frame = Frame()
        confocal_frame['a1'] = e2[1]-e2[0]
        confocal_frame['a2'] = e1[1]-e2[0]
        confocal_frame['p'] = e2[0]
        self.construct(muscles,confocal_frame)

    def coords_from_frame(self,other_frame):
        ret = dict()
        for key in self.barycentric.keys():
            coords = np.dot(other_frame['A'],(self.barycentric[key]).T) + \
                     other_frame['p'][:,np.newaxis]
            ret[key] = coords
        return(ret)
    
    def get_masks(self,fly_frame,(sizex,sizey)):
        from matplotlib.path import Path
        plot_lines = self.coords_from_frame(fly_frame)
        masks = dict()
        for line_key in plot_lines.keys():
            plot_line = plot_lines[line_key]
            p = Path(plot_line.T)
            xpnts,ypnts = np.meshgrid(range(sizex),range(sizey))
            testpnts = np.vstack((xpnts.ravel(),ypnts.ravel()))
            masks[line_key] = p.contains_points(testpnts.T).reshape(sizey,sizex)
        return masks
    
    def get_masks_by_line(self,fly_frame,(sizex,sizey),line_name = None,muscles = None):
        #get the mask of all the muscles for fit
        masks = self.get_masks(fly_frame,(sizex,sizey))
        #create the model using only the muscles that express in a given line
        if muscles == None:
            muscles = get_muscle_list(line_name)
            muscles = [m for m in muscles if not('DVM' in m) and not('DLM' in m) and not('ps' in m)]
        model = np.vstack([masks[mask_key].T.ravel().astype(float) for mask_key in muscles])
        fit_pix_mask = np.sum(model,axis=0) > 0
        #add a background term
        model = np.vstack([model,np.ones_like(masks[mask_key].ravel())])
        return model,fit_pix_mask
        
class ModelView(object):
    def __init__(self,model):
        import copy
        self.model = model
        self.plot_frame = copy.copy(model.frame)
        #self.plot_basis['p'] = default_rframe_data['p']
        #self.plot_basis['a1'] = default_rframe_data['a1']
        #self.plot_basis['a2'] = default_rframe_data['a2']

    def plot(self,plot_frame,plotobject):
        lines = self.model.coords_from_frame(plot_frame)
        self.curves = list()
        for line in lines.values():
            self.curves.append(plotobject.plot(line[0,:],line[1,:]))

    def update_frame(self,plot_frame):
        lines = self.model.coords_from_frame(plot_frame)
        for curve,line in zip(self.curves,lines.values()):
            curve.setData(line[0,:],line[1,:])

    def frame_changed(self,roi):
        pnts = roi.saveState()['points']
        p = np.array(pnts[1])

        a1 = np.array(pnts[0])-p
        a2 = np.array(pnts[2])-p

        self.plot_frame['p'] = p
        self.plot_frame['a1'] = a1
        self.plot_frame['a2'] = a2
        self.update_frame(self.plot_frame)

class ModelViewMPL(ModelView):

    def plot(self,plot_frame,**kwargs):
        import pylab as plb
        default_args = {'draw_frame':True,
                        'frame_head_width':20,
                        'contour_kwargs':{'edgecolor': 'none', 
                                          'linewidth': 0.0, 
                                          'facecolor': 'none'}}
        from pylab import plot,arrow
        lines = self.model.coords_from_frame(plot_frame)
        self.curves = list()
        #plot_args = kwargs.pop('plot_args',default_args)
        for line_key in lines.keys():
            try:
                element_args = kwargs['contour_kwargs'][line_key]
            except KeyError:
                
                element_args = default_args['contour_kwargs']
            line = lines[line_key]
            from matplotlib.patches import Polygon
            poly = Polygon(zip(line[0,:],line[1,:]),**element_args)
            #plb.plot([1,2,3,4])
            plb.gca().add_patch(poly,)

        if 'draw_frame' in kwargs.keys():
            if kwargs['draw_frame']:
                frame_args = dict()
                p = plot_frame['p']
                a1 = plot_frame['a1']
                a2 = plot_frame['a2']
                frame_args['color'] = 'g'
                frame_args['head_width'] = kwargs['frame_head_width']
                arrow(p[0],p[1],a1[0],a1[1],**frame_args)
                frame_args['color'] = 'b'
                frame_args['head_width'] = kwargs['frame_head_width']
                arrow(p[0],p[1],a2[0],a2[1],**frame_args)

def warp_fly_img(netfly = None,img = None,s = 1):        
    #import muscle_model as mm
    import group_data as gd
    confocal_model = GeometricModel(filepath = gd.muscle_anatomy_dir + 'confocal_outline_model.cpkl')
    confocal_view = ModelViewMPL(confocal_model)
    import cv2
    import cPickle
    pkname = netfly.fly_path + '/basis_fits.cpkl'
    fly_frame = Frame()
    fly_frame.load(pkname)
    A = confocal_model.frame.get_transform(fly_frame)
    #compose the transformation matrix with a scaling matrix
    Ap = np.dot([[s,0.0,0],[0,s,0],[0,0,1]],A)
    output_shape = (np.array([1024,1024])*s).astype(int) #confocal shape * the scale
    output_shape = (output_shape[0],output_shape[1]) #make the shape a tuple
    X_warped = cv2.warpAffine(img.T,Ap[:-1,:],output_shape) #warp the image using the cv2 warp Affine method
    return X_warped
