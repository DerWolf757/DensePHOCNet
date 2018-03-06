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
        
        p_p = p1-p2
        p1_n = p1 - n
        p2_n = p2 - n
        
        pp = np.dot(p_p,p_p)
        p1n = np.dot(p1_n,p1_n)
        p2n = np.dot(p2_n,p2_n)
        
        pn = p1n if p1n < p2n else p2n
        
        exp_pp = np.exp(pp)
        exp_pn = np.exp(pn)
        
        self.loss = (exp_pp/(exp_pn+exp_pp))**2 +(exp_pn/(exp_pn+exp_pp)-1)**2
        
        top[0].data[...] = self.loss
        
        
        
        
        
      
        
        

         

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.diff
        bottom[1].diff[...] = self.diff
        bottom[2].diff[...] = self.diff

        
        