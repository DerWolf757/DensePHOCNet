'''
Created on Jan 16, 2018

@author: fwolf

'''
import numpy as np
from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.io.files import save_prototxt

def main():
    file_path = '/home/fwolf/Workspace/DensePHOCNet/src/example/created.prototxt'
    triplet_string_file = '/home/fwolf/Workspace/DensePHOCNet/data/depth_evaluation/proto/train_dense_L50b2k32tpp_phocnet__iam-db.prototxt'
    train_word_images_lmdb_path = '/data/fwolf/ordered_iam_50000/iam-db_nti500000_pul2-3-4-5_train_word_images_lmdb'
    
    mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=0)
    proto = mpg.get_triplet_net(train_word_images_lmdb_path = train_word_images_lmdb_path)
    
    save_prototxt(file_path='/home/fwolf/Workspace/DensePHOCNet/src/example/test.prototxt', proto_object=proto, header_comment='Solver PHOCNet')
    '''
    with open(file_path, 'w') as output_file:
        output_file.write(proto)
    '''

if __name__ == '__main__':
    main()