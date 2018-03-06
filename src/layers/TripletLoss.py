'''
Created on Dec 14, 2017

@author: fwolf
'''

import caffe
import numpy as np
from scipy.spatial.distance import cdist


class TripletLossLayer(caffe.Layer):
    """
    Permutes the bottom blob such that neurons of early feature maps are next to late ones.
    """

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need one input to compute distance.")
        
        if len(top) != 1:
            raise Exception("Need one output.")
        
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        p1 = bottom[0].data
        p2 = bottom[1].data
        n = bottom[2].data
        
        p1 = p1/np.linalg.norm(p1)
        p2 = p1/np.linalg.norm(p2)
        n = p1/np.linalg.norm(n)
        
        a_p = (p1-p2).flatten()
        a_n = (p1-n).flatten()
        
        ap = np.dot(a_p,a_p)
        an = np.dot(a_n,a_n)
        
        dist = (0.2 + ap - an)
        self.loss = max(dist,0.0)

        top[0].data[...] = self.loss


    def backward(self, top, propagate_down, bottom):
        x_a = bottom[0].data
        x_p = bottom[1].data
        x_n = bottom[2].data
        
        #print x_a,x_p,x_n
        bottom[0].diff[...] =  (x_n - x_p)/((bottom[0]).num)
        bottom[1].diff[...] =  (x_p - x_a)/((bottom[0]).num)
        bottom[2].diff[...] =  (x_a - x_n)/((bottom[0]).num)


        
        