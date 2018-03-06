'''
Created on Nov 20, 2017

@author: fwolf
'''

import os

from caffe import NetSpec
from phocnet.caffe.model_proto_generator import ModelProtoGenerator
from phocnet.io.files import save_prototxt, write_list


def main():
    mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=0)
    train_word_images_lmdb_path = '/data/fwolf/min26/gw_cv1_nti500000_pul2-3-4-5_train_word_images_lmdb'
    train_phoc_lmdb_path = '/data/fwolf/min26/gw_cv1_nti500000_pul2-3-4-5_train_phocs_lmdb'
    
    phoc_size = 604
    dense_net_file = None
    
    nblocks = 2
    growth_rate = 12
    nlayers = 65
    config = (20,40)
    use_bottleneck = False
    use_compression = True
    pool_init = True
    conv_init = (64,64)
    no_batch_normalization = False
    pooling = 'tpp'
    init_7 = False
    
    dropout_ratio = 0.0
    train_word_images_lmdb_path=''
    
    
    train_proto = mpg.get_dense_phocnet(train_word_images_lmdb_path, train_phoc_lmdb_path, phoc_size, dense_net_file,
                                  pooling, nblocks, growth_rate, nlayers, config, pool_init, conv_init, init_7,
                                  no_batch_normalization,
                                  use_bottleneck, use_compression, dropout_ratio, max_out=-1, use_perm=False,
                                  generate_deploy=False)



    #dense_net_file = '/home/fwolf/Workspace/DensePHOCNet/src/example/DenseNet_121.prototxt'
    
    #n = NetSpec()
    #train_proto = mpg.get_triplet_net(train_word_images_lmdb_path, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio)

    '''
    
    str_B = 'B' if use_bottleneck else ''
    str_C = 'C' if use_compression else ''
    str_BC = ''
    
    if (str_B + str_C):
        str_BC = '_'+str_B+str_C
    
    name = ('dense_phocnet_L%ib%ik%i' % (nlayers,nblocks,growth_rate))+str_BC
    '''
    name = 'ConvInitTest'

    file_path = '../example/' + name +'.prototxt'

    save_prototxt(file_path=os.path.join(file_path), proto_object=train_proto, header_comment='Train PHOCNet')

if __name__ == '__main__':
    main()