'''
Created on Nov 24, 2017

@author: fwolf
'''
import logging
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

from phocnet.evaluation.retrieval import map_from_feature_matrix

from skimage.transform import resize

def main():
    logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=logging_format)
    logger = logging.getLogger('feature_db')
    logger.info('logger loaded...')
    
    lmdb_file = feature_db = '/data/fwolf/feature_db/test_L50b2k32'
    caffe.set_mode_gpu()
    caffe.set_device(1)
    
    
    train_annotation_file = "/home/fwolf/Workspace/DensePHOCNet/experiments/iam/train.xml"
    test_annotation_file = "/home/fwolf/Workspace/DensePHOCNet/experiments/iam/test.xml"
    doc_img_dir = "/vol/corpora/document-image-analysis/iam-db/images/"
    
    xml_reader = XMLReader(make_lower_case=False)
    
    dataset_name, train_list, test_list = xml_reader.load_train_test_xml(train_xml_path=train_annotation_file, 
                                                                                  test_xml_path=test_annotation_file, 
                                                                                  img_dir=doc_img_dir)
    
    
    
    fc_net_file = '/home/fwolf/Workspace/DensePHOCNet/data/triplet/fc_deploy.prototxt'
    weights =  '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/models/dense_L50b2k32tpp_phocnet__iam-db_nti500000_pul2-3-4-5.binaryproto'
    
    fc_net = caffe.Net(fc_net_file, caffe.TEST)
    fc_net.copy_from(weights)
    
    lmdb_env = lmdb.open(lmdb_file, readonly=True) 
    
    
    
    output = []
    counter = 0
    
    
    with lmdb_env.begin() as lmdb_txn:
        lmdb_cursor = lmdb_txn.cursor()
        while lmdb_cursor.next():    
            # generate phoc from fc net
            value = lmdb_cursor.value()
            key = lmdb_cursor.key()
            
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            
            
            feat = caffe.io.datum_to_array(datum)
            
            fc_net.blobs['features'].data[...] = feat[0,:,:]
            fc_net.forward()
            
            phoc = fc_net.blobs['sigmoid'].data.flatten()

            output.append(phoc)
            
            counter += 1
            if counter % 100 == 0:
                print '   Finished %i ndarrays' % counter

                    
                
       
    logger.info('Predicting PHOCs for %d test words', len(test_list))

    logger.info('Calculating mAP...')
    
    phocs = np.vstack(output)
    _, avg_precs = map_from_feature_matrix(features=phocs, labels=[word.get_transcription() for word in test_list], 
                                                   metric='cosine', drop_first=True)
    logger.info('mAP: %f', np.mean(avg_precs[avg_precs > 0])*100)                

    
 


def printPHOC(phoc):
    print 'First Split:\n'
    split1 = phoc[:37]
    ids = np.nonzero(split1)
    
    string = '';
    for k in ids[0]:
        string = string + '"' + str(unichr((k-11)+97)) +  '"     '
                          
    print string
    
    print 'Second Split:\n'
    split2 = phoc[37:75]
    ids = np.nonzero(split2)
    
    string = '';
    for k in ids[0]:
        string = string + '"' + str(unichr((k-11)+97)) +  '"     '
    print string


    

if __name__ == '__main__':
    main()