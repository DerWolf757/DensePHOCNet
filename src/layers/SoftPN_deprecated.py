'''
Created on Dec 14, 2017

@author: fwolf
'''

import caffe
import numpy as np
from scipy.spatial.distance import cdist


class SoftPNLossLayer(caffe.Layer):
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
        
        
        # normalize p1
        p1 = p1/np.linalg.norm(p1)

        
        # normalize p2
        p2 = p2/np.linalg.norm(p2)
        
        # normalize n
        n = n/np.linalg.norm(n)
        
        
        
        
        p1_p2 = np.linalg.norm(p1-p2)
        p1_n = np.linalg.norm(p1-n)
        p2_n = np.linalg.norm(p2-n)

        
        min_d = np.min((p1_n,p2_n))
        exp_min = np.exp(min_d)
        exp_p = np.exp(p1_p2)
        
        denominator = exp_min + exp_p
        
        a = exp_p/denominator
        b = (exp_min/denominator)
        b = b-1
        
        a2 = a**2
        b2 = b**2
        
        self.loss = a2 + b2
        top[0].data[...] = self.loss
        

         

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.loss
        bottom[1].diff[...] = self.loss
        bottom[2].diff[...] = self.loss

        
        