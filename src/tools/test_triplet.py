'''
Created on Nov 20, 2017

@author: fwolf
'''

import caffe

from caffe import NetSpec
from caffe import layers as L
import numpy as np
import os
import phocnet
import cv2
from caffe.proto import caffe_pb2

import lmdb

from skimage.transform import resize
from wordspotting import gt_reader

from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.io.files import save_prototxt, write_list

from layers import triplet



def main():
    mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=0)
    
    nblocks = 4
    growth_rate = 32
    nlayers = 121
    config = (6,12,24,16)
    use_bottleneck = True
    use_compression = True
    pool_init = True
    no_batch_normalization = False
    
    dropout_ratio = 0.0
    train_word_images_lmdb_path = '/data/fwolf/ordered_iam_50000/iam-db_nti500000_pul2-3-4-5_train_word_images_lmdb'
    
    n = NetSpec()
    
    train_proto = mpg.get_triplet_net(train_word_images_lmdb_path, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio)
    name = 'triplet'
    file_path = '/home/fwolf/Workspace/DensePHOCNet/src/example/' + name +'.prototxt'
    save_prototxt(file_path=os.path.join(file_path), proto_object=train_proto, header_comment='Train PHOCNet')
    
    
    fc_net = caffe.Net(file_path, caffe.TEST)

    




    
    
if __name__ == '__main__':
    main()