'''
Created on Nov 24, 2017

@author: fwolf
'''
import logging
import argparse

import lmdb
import caffe

from caffe import NetSpec
from caffe import layers as L


import cv2
import numpy as np

from phocnet.io.xml_io import XMLReader
from caffe.proto import caffe_pb2
from phocnet.evaluation.retrieval import map_from_feature_matrix

from phocnet.io.files import save_prototxt


def main():
    logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=logging_format)
    logger = logging.getLogger('feature_db')
    

    
    parser = argparse.ArgumentParser()
    # required training parameters
    parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                      help='The location of the document images.')
    parser.add_argument('--train_annotation_file', action='store', type=str, required=True,
                      help='The file path to the READ-style XML file for the training partition of the dataset to be used.')
    parser.add_argument('--test_annotation_file', action='store', type=str, required=True,
                      help='The file path to the READ-style XML file for the testing partition of the dataset to be used.')
    parser.add_argument('--lmdb_dir', action='store', type=str, required=True,
                      help='Directory where to to find the LMDB databases.')
    
    parser.add_argument('--weights', action='store', type=str,
                      help='The location of the dense net weights.')
    
    parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default = 0,
                      help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    
    
    params = vars(parser.parse_args())

    
    #lmdb_file = feature_db = '/data/fwolf/feature_db/test_L50b2k32'
    caffe.set_mode_gpu()
    caffe.set_device(params["gpu_id"])
    
    lmdb_dir = params["lmdb_dir"]
    train_annotation_file = params["train_annotation_file"]
    test_annotation_file = params["test_annotation_file"] #"/home/fwolf/Workspace/DensePHOCNet/experiments/iam/test.xml"
    doc_img_dir = params["doc_img_dir"] #"/vol/corpora/document-image-analysis/iam-db/images/"
    
    weights = params["weights"]
    
    xml_reader = XMLReader(make_lower_case=False)
    
    dataset_name, train_list, test_list = xml_reader.load_train_test_xml(train_xml_path=train_annotation_file, 
                                                                                  test_xml_path=test_annotation_file, 
                                                                                  img_dir=doc_img_dir)
    
    
    
    
    
    n = NetSpec()
    n.features = L.Input(shape=dict(dim=[1, 22080]))
    
    n.fc6_d_p1 = L.InnerProduct(n.features, num_output=4096, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    n.relu6_p1 = L.ReLU(n.fc6_d_p1, in_place=True)
    
    n.fc7_d_p1 = L.InnerProduct(n.relu6_p1, num_output=4096, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    n.relu7_p1 = L.ReLU(n.fc7_d_p1, in_place=True)

    n.fc8_d_p1 = L.InnerProduct(n.relu7_p1, num_output=618, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    n.sigmoid = L.Sigmoid(n.fc8_d_p1)
    
    
    
    save_prototxt(file_path='fc_deploy.prototxt', proto_object=n.to_proto(), header_comment='Evaluate FC Part')
    
    fc_net = caffe.Net('fc_deploy.prototxt', caffe.TEST)
    fc_net.copy_from(weights)
    
    
    lmdb_env = lmdb.open(lmdb_dir, readonly=True) 
    
    
    
    output = []
    counter = 0
    
    logger.info('Predicting PHOCs for %d test words', len(test_list))
    
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
                logger.info('   Finished %i ndarrays' % counter)

                    
                
       

    logger.info('Calculating mAP...')
    
    phocs = np.vstack(output)
    _, avg_precs = map_from_feature_matrix(features=phocs, labels=[word.get_transcription() for word in test_list], 
                                                   metric='cosine', drop_first=True)
    logger.info('mAP: %f', np.mean(avg_precs[avg_precs > 0])*100)                


if __name__ == '__main__':
    main()