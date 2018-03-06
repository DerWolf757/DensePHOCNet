'''
Created on Dec 14, 2017

@author: fwolf
'''

import caffe
import numpy as np


class PermuteLayer(caffe.Layer):
    """
    Permutes the bottom blob such that neurons of early feature maps are next to late ones.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Need one input to compute distance.")
        
        if len(top) != 1:
            raise Exception("Need one output.")
        
    def reshape(self, bottom, top):
        # Copy shape from bottom
        top[0].reshape(0)

    def forward(self, bottom, top):
        params = eval(self.param_str)
        
        fm_size = params["fm_size"]
        k = params["k"]
        
        bottom_data = bottom[0].data.copy()
        
        # Number of feature maps
        fm_count = np.shape(bottom_data)[1]/fm_size
        
        # Reshape feature maps. Each row represents one feature map
        feature_mat = np.reshape(bottom_data,(fm_count,fm_size))
  
        # Transpose. Each column represents one feature map
        feature_transposed = np.transpose(feature_mat)

        # Rearrange columns/feature maps
        # idx: [0 1 2 3 4 5 6 7 8 9] --> [0 9 1 8 2 7 3 6 4 5]
        idx = np.zeros(fm_count)
        if k == 1:
            idx[::k] = range(0,fm_count/2,1)
            idx[1::k] = range(fm_count-1,fm_count/2-1,-1)
            
        for i in range(k/2):
            idx[i::k] = range(i,fm_count/2,k/2)
            idx[k/2+i::k] = range(fm_count-1-i,fm_count/2-1,-k/2)
        
        feature_rearranged = feature_transposed[:,idx.astype(int)]
           
        # Reshape into vector
        top[0].data[...] = np.reshape(feature_rearranged,np.shape(bottom_data))


    def backward(self, top, propagate_down, bottom):
        params = eval(self.param_str)
        
        fm_size = params["fm_size"]
        k = params["k"]

        top_data = top[0].data.copy()
        
        # Number of feature maps
        fm_count = np.shape(top_data)[1]/fm_size
        
        # Reshape feature maps. Each column represents one feature map
        feature_rearranged = np.reshape(top_data,(fm_size,fm_count))
        
        
        # Rearrange columns/feature maps
        # idx: [0 1 2 3 4 5 6 7 8 9] --> [0 2 4 6 8 9 7 5 3 1]
        idx = np.zeros(fm_count)
        if k == 1:
            idx[:fm_count/2:1] = range(0,fm_count,2)
            idx[fm_count/2::1] = range(fm_count-1,0,-2)
        
        for i in range(k/2):
            idx[i:fm_count/2:k/2] = range(i,fm_count,k)
            idx[fm_count/2+i::k/2] = range(fm_count-1-i,0,-k)
        
        
        feature_transposed = feature_rearranged[:,idx.astype(int)]
        
        
        # Transpose. Each row represents one feature map
        feature_mat = np.transpose(feature_transposed)      
        
        # Reshape into vector
        bottom_data = np.reshape(feature_mat,np.shape(top_data))
    
        bottom[0].data[...] = bottom_data
        
        