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


def main():
    lmdb_file = "/data/fwolf/iam_max/iam-db_nti500000_pul2-3-4-5_test_word_images_lmdb"
    
    #train_annotation_file = "../../experiments/gw/gw_cv1_train.xml"
    #test_annotation_file = "../../experiments/gw/gw_cv1_test.xml"
    
    train_annotation_file = "../../experiments/iam/train.xml"
    test_annotation_file = "../../experiments/iam/test.xml"
    doc_img_dir = "/vol/corpora/document-image-analysis/gw/pages/"
    
    xml_reader = XMLReader(make_lower_case=False)
    
    dataset_name, train_list, test_list = xml_reader.load_train_test_xml(train_xml_path=train_annotation_file, 
                                                                                  test_xml_path=test_annotation_file, 
                                                                                  img_dir=doc_img_dir)
    #sorted(train_list, key=lambda transcription: train_list.get_transcription)
    
    train_list.sort(key=operator.methodcaller("get_transcription"))
  
    
    size = np.zeros(len(train_list))
    width = np.zeros(len(train_list))
    
    k = 0
    
    '''
    for item in train_list:
        bb = item.get_bounding_box()
        size[k] = bb['widthHeight'][0]*bb['widthHeight'][1]
        width[k] = bb['widthHeight'][0]
        k = k+1
    
    bb_vec = [b for b in train_list if b.get_bounding_box()['widthHeight'][0]>580 and b.get_bounding_box()['widthHeight'][1] >= 170]
    for b in bb_vec:
        print b.get_bounding_box()
        
    print len(bb_vec)

    i = np.argmax(size)
    print size[i]
    
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
                print train_list[idx[0][0]].get_bounding_box()
            
    
    
    print count_vec
    print sum(count_vec[12:])
    print len(train_list)
    '''
    
        

    
    lmdb_env = lmdb.open(lmdb_file, readonly=True) 
    
    min0 = (10000,10000) 
    id1 = -1
    
    min1 = (10000,10000) 
    id2 = -1
    max0 = (0,0)
    max1 = (0,0)
    
    avg = np.zeros(2)
    
    k = 0
   
    counter = 0
    with lmdb_env.begin() as lmdb_txn :
        lmdb_cursor = lmdb_txn.cursor() 
        length = lmdb_txn.stat()['entries']
        print 'db_size: %i' % length
        #for it in lmdb_cursor.iternext() :
        while lmdb_cursor.next() :
            value = lmdb_cursor.value()
            key = lmdb_cursor.key()

            
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            image = np.zeros((datum.channels, datum.height, datum.width))
            image = caffe.io.datum_to_array(datum)   
            image = np.transpose(image, (1, 2, 0))    
            image = image[:,:,::-1]
            
            avg = avg + image.shape[:2]
            
            if image.shape[0] < min0[0]: min0 = image.shape
            if image.shape[1] < min1[1]: min1 = image.shape
            
            if image.shape[0] > max0[0]: 
                max0 = image.shape
                id1 = value
            if image.shape[1] > max1[1]: 
                max1 = image.shape
                id2 = value
            
            counter = counter +1
            
            if counter % 10000 == 0:
                print '%i images processed' % counter
                
            
            

            
            
             
            '''
            print("key: ", key) 
            print("image shape: " + str(image.shape) + ", data type: " + str(image.dtype) + ", random pixel value: " +  str(image[10,10,0]))
                        
            cv2.imshow('img1',image)
            
            cv2.waitKey(100000)
            '''
                
    datum = caffe_pb2.Datum()
    datum.ParseFromString(id2)
    image = np.zeros((datum.channels, datum.height, datum.width))
    image = caffe.io.datum_to_array(datum)   
    image = np.transpose(image, (1, 2, 0))    
    image = image[:,:,::-1]
    
    cv2.imshow('img1',image)
            
    print min0, max0
    print min1, max1
    print 'Average: %ix%i ' % (avg[0]/length, avg[1]/length)
    print image.shape
    
    cv2.waitKey(100000)
    
    
    
    
    
    
    
    '''
    min1 = 10000; id1 = -1;
    min2 = 10000; id2 = -1;
    max1 = 0;
    max2 = 0;
    
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
    
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        
        if np.shape(data)[1] < min1: 
            min1 = np.shape(data)[1] 
            id1 = value
            
        if np.shape(data)[2] < min2: 
            min2 = np.shape(data)[2]
            id2 = value
            
        if np.shape(data)[1] > max1: max1 = np.shape(data)[1]
        
        if np.shape(data)[2] > max2: max2 = np.shape(data)[2]
        
        
    datum.ParseFromString(id1);
    data1 = caffe.io.datum_to_array(datum)
    data1 = np.transpose(data1,(2,1,0))
    
    print np.shape(data1)
    
    datum.ParseFromString(id2)
    data2 = caffe.io.datum_to_array(datum)
    data2 = np.transpose(data2,(2,1,0))
    
    print np.shape(data2)
    
    cv2.imshow('min1',data1)
    cv2.imshow('min2',data2)
    
    cv2.waitKey(100000)
    '''





    

if __name__ == '__main__':
    main()