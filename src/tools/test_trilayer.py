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
    

    
    fc_net_path = '/home/fwolf/Workspace/DensePHOCNet/data/triplet/train_tripletdense_L50b2k32tpp_phocnet__iam-db.prototxt'
    fc_net = caffe.Net(fc_net_path, caffe.TEST)
    
    weights =  '/home/fwolf/Workspace/DensePHOCNet/data/triplet/tripletdense_L50b2k32tpp_phocnet__iam-db_nti500000_pul2-3-4-5_it100.binaryproto'

    
    base_weights =  '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/models/dense_L50b2k32tpp_phocnet__iam-db_nti500000_pul2-3-4-5.binaryproto'
    base_net_path = '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/deploy/L50b2k32iam-db.prototxt'


    print 'Initializing Weights'
    base_net_path = '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/proto/train_dense_L50b2k32tpp_phocnet__iam-db.prototxt'
    base_net = caffe.Net(base_net_path, caffe.TEST)
    base_net.copy_from(base_weights)

            
            
    print 'Initializing fc network...'
    param_fc6 = base_net.params["fc6_d"]
    for k in range(len(param_fc6)):
        fc_net.params['fc6_d_p1'][k].data[...] = param_fc6[k].data
        fc_net.params['fc6_d_p2'][k].data[...] = param_fc6[k].data
        fc_net.params['fc6_d_n'][k].data[...] = param_fc6[k].data
        
    param_fc7 = base_net.params["fc7_d"]
    for k in range(len(param_fc7)):
        fc_net.params['fc7_d_p1'][k].data[...] = param_fc7[k].data
        fc_net.params['fc7_d_p2'][k].data[...] = param_fc7[k].data
        fc_net.params['fc7_d_n'][k].data[...] = param_fc7[k].data
        
    param_fc8 = base_net.params["fc8_d"]
    for k in range(len(param_fc8)):
        fc_net.params['fc8_d_p1'][k].data[...] = param_fc8[k].data
        fc_net.params['fc8_d_p2'][k].data[...] = param_fc8[k].data
        fc_net.params['fc8_d_n'][k].data[...] = param_fc8[k].data
    
    
    
    
    
    
    for i in range(100):
        fc_net.forward()

        

        
        
        
        


        
        
        
        


    
    
if __name__ == '__main__':
    main()