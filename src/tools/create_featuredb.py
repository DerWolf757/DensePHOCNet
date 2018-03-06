'''
Created on Jan 29, 2018

@author: fwolf
'''
import logging
import argparse
import numpy as np

import caffe
import lmdb
import cv2
from caffe.proto import caffe_pb2


def main():
    logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=logging_format)
    logger = logging.getLogger('feature_db')
    logger.info('logger loaded...')
    
    caffe.set_mode_gpu()
    caffe.set_device(2)
    
    lmdb_dir = '/data/fwolf/ordered_iam_50000/iam-db_nti500000_pul2-3-4-5_train_word_images_lmdb'
    feature_db = '/data/fwolf/feature_db/train_L50b2k32'
    
    net_file = '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/deploy/L50b2k32iam-db.prototxt'
    weights =  '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/models/dense_L50b2k32tpp_phocnet__iam-db_nti500000_pul2-3-4-5.binaryproto'
    
    print 'Loading network....'
    
    net = caffe.Net(net_file, caffe.TEST)
    net.copy_from(weights)
    
    logger.info('Loading network done!!!')
    

    logger.info('Opening single LMDB at %s for writing' % feature_db)
    
    database_images = lmdb.open(path=feature_db, map_size=1024**4)
    txn_images = database_images.begin(write=True)
    counter = 0
    
    
    lmdb_env = lmdb.open(lmdb_dir, readonly=True)
    with lmdb_env.begin() as lmdb_txn :
        lmdb_cursor = lmdb_txn.cursor() 
        while lmdb_cursor.next():
            value = lmdb_cursor.value()
            key = lmdb_cursor.key()
            
            
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            datum_label = datum.label
            datum = caffe.io.datum_to_array(datum)
            datum = datum.astype(float)
            datum -= 255.0
            datum /= -255.0
            
            
            datum = np.reshape(datum, (1,1,np.shape(datum)[1], np.shape(datum)[2]))

            net.blobs["word_images"].reshape(*datum.shape)
            net.reshape()
            
            net.blobs["word_images"].data[...] = datum
            net.forward()
            
            feat2 = net.blobs["tpp5"].data

            # convert img_mat to Caffe Datum
            feat2 = np.reshape(feat2, (1,1,np.shape(feat2)[1]))
            datum = caffe.io.array_to_datum(arr=feat2, label=datum_label)

            txn_images.put(key=key, value=datum.SerializeToString())
            counter += 1
            if counter % 100 == 0:
                txn_images.commit()
                logger.info('   Finished %i ndarrays' % counter)
                # after a commit the txn object becomes invalid, so we need to get a new one
                txn_images = database_images.begin(write=True)
                
    
    logger.info('   Finished %i ndarrays' % counter)     
    txn_images.commit()
    database_images.sync()
    database_images.close()
    

    
    
    

if __name__ == '__main__':
    main()