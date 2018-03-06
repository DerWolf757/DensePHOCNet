'''
Created on Nov 20, 2017

@author: fwolf
'''

import caffe
import numpy as np
import os
import phocnet
import cv2
import permute

from skimage.transform import resize
from wordspotting import gt_reader



def main():
    caffe.set_mode_gpu()
    caffe.set_device(2)

    net_path = '/home/fwolf/Workspace/DensePHOCNet/src/example/python_layer.prototxt'

    net = caffe.Net(net_path, caffe.TEST)
    
    L = 30

    input = np.arange(L).reshape((1,L))
    net.blobs['in'].data[...] = input
    
    print net.blobs['in'].data[:L]
    
    net.forward()
    
    print net.blobs['out'].data[:L]
    
    backward(net.blobs['out'].data,net.blobs['in'].data)
    
def backward(top, bottom):
        fm_size = 5
        
        top_data = top
        fm_count = np.shape(top_data)[1]/fm_size
        
        feature_rearranged = np.reshape(top_data,(fm_size,fm_count))
        
        idx = np.zeros(fm_count)
        idx[:fm_count/2] = range(0,fm_count,2)
        idx[fm_count/2:] = range(fm_count-1,0,-2)
        
        
        feature_transposed = feature_rearranged[:,idx.astype(int)]
        
        feature_mat = np.transpose(feature_transposed)
  
        out = np.reshape(feature_mat,np.shape(top_data))
        
        print out
        
        
        
        


    
    
if __name__ == '__main__':
    main()