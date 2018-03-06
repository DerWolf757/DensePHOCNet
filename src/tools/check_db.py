'''
Created on Nov 24, 2017

@author: fwolf
'''
import numpy as np
import lmdb
import caffe
import cv2
from caffe.proto import caffe_pb2
import operator


from phocnet.attributes.phoc import build_phoc, unigrams_from_word_list,\
    get_most_common_n_grams
from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.caffe.solver_proto_generator import generate_solver_proto
from phocnet.caffe.lmdb_creator import CaffeLMDBCreator
from phocnet.caffe.augmentation import AugmentationCreator
from phocnet.evaluation.time import convert_secs2HHMMSS
from phocnet.evaluation.cnn import calc_map_from_cnn_features
from phocnet.io.xml_io import XMLReader
from phocnet.io.files import save_prototxt, write_list
from phocnet.io.context_manager import Suppressor
from phocnet.numpy.numpy_helper import NumpyHelper

from ws_seg_based.wordspotting_tools.dataset_loader import DatasetLoader

from skimage.transform import resize

def main():

    
    train_list, test_list, qry_list = DatasetLoader.load_icfhr2016_competition('botany',
                                                                                train_set='Train_III',
                                                                                path='/vol/corpora/document-image-analysis/competition_icfhr2016/')
    

    bb_vec = [b for b in train_list]
    #for b in bb_vec:
    #    print b.get_bounding_box()
    
    size = np.zeros(len(train_list))
    width = np.zeros(len(train_list))
    
    min = (2000, 1000)
    
    k = 0
    rescales = 0
    for item in train_list:
        bb = item.get_bounding_box()
        size[k] = bb['widthHeight'][0]*bb['widthHeight'][1]
        width[k] = bb['widthHeight'][0]
        k = k+1
        
        if bb['widthHeight'][0]*bb['widthHeight'][1] > 50000:
            img = item.get_word_image(gray_scale=True)
            
            cv2.imshow('Original',img)
            
            new_img, res  = __check_size(img)
            
            cv2.imshow('Resized',new_img)
            cv2.waitKey(100000)
            
            rescales +=1
        
        
    print 'number of rescales: %i' % rescales
    print 'database size: %i' % len(train_list)

    i = np.argmax(size)
    print 'size of biggest element: %i' % size[i]
    
    # get statistics
    count_vec = np.zeros(23)
    k = 0
    for step in range(0,230000,10000):
        vec = [s for s in size if s>=step and s < step+10000]
        count_vec[k] = len(vec)
        k = k+1
        if step >= 120000:
            for item in vec:
                idx = np.where(size==item)
                a = idx[0]
                #print train_list[idx[0][0]].get_bounding_box()
            
    
    
    #print count_vec
    #print sum(count_vec)
    #print len(train_list)
    
def __check_size(img):
        '''
        checks if the image accords to the minimum and maximum size requirements
        
        Returns:
            tuple (img, bool):
                 img: the original image if the image size was ok, a resized image otherwise
                 bool: flag indicating whether the image was resized
        '''
        min_image_height = 32
        min_image_width = 32
        max_pixel = 50000
        
        # check minimal size
        scale_height = float(min_image_height+1)/float(img.shape[0])
        scale_width = float(min_image_width+1)/float(img.shape[1])
        
        if img.shape[0] < min_image_height and scale_height*img.shape[1] > min_image_width:
            resized = True
            new_shape = (int(scale_height*img.shape[0]), int(scale_height*img.shape[1]))
        elif img.shape[1] < min_image_width and scale_width*img.shape[0] > min_image_height:
            resized = True
            new_shape = (int(scale_width*img.shape[0]), int(scale_width*img.shape[1]))
        else:
            resized = False
            
            
        # check maximum image size
        if img.shape[0]*img.shape[1] > max_pixel:
            resized = True
            relation = float(img.shape[0])/float(img.shape[1])
            
            i0 = np.sqrt(max_pixel/relation)
            i1 = i0*relation
            
            new_shape = (int(i1), int(i0))

        
        if resized:
            new_img = resize(image=img, output_shape=new_shape)
            new_img = (new_img*255).astype('uint8')
            
            return new_img, resized
        else:
            return img, resized
    


if __name__ == '__main__':
    main()