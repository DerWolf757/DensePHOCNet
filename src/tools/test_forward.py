'''
Created on Nov 20, 2017

@author: fwolf
'''

import caffe
import numpy as np
import os
import phocnet
import cv2

from skimage.transform import resize
from wordspotting import gt_reader

def load_image(doc_id, word_id):
    gw_path = '/vol/corpora/document-image-analysis/gw/'
    gt_r = gt_reader.GroundTruthReader(gw_path,gt_dir = 'ground_truth/')
    
    file_names = sorted(os.listdir(gt_r.get_base_path()))
    
    gt_list = gt_r.read_ground_truth(os.path.splitext(file_names[doc_id])[0])

    if(len(gt_list) < 1):
            raise ValueError('gt_list is zero!', len(gt_list))
        
    page = cv2.imread(gw_path + 'pages/' + os.path.splitext(file_names[doc_id])[0] + '.png',cv2.CV_LOAD_IMAGE_GRAYSCALE) 
    
    bbox = gt_list[word_id][1];
    
    word = page[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
    
    
    string = gt_list[word_id][0]
    
    return word, string


def main():
    caffe.set_mode_gpu()
    caffe.set_device(1)

    net_path = '/home/fwolf/Workspace/DensePHOCNet/src/example/train_fftpp_phocnet__iam-db.prototxt'
    #net_path = '/home/fwolf/Workspace/DensePHOCNet/src/example/perm_test.prototxt'

    phocnet = caffe.Net(net_path, caffe.TEST)
    
    phocnet.forward()
    
    image = phocnet.blobs['word_images'].data
    
    print image.shape

    


if __name__ == '__main__':
    main()