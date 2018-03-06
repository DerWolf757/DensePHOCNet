# pylint: disable=too-many-arguments
'''
Created on Jul 8, 2016

@author: ssudholt
'''
import logging

from caffe import NetSpec
from caffe import layers as L
from caffe import params as P
from caffe.io import caffe_pb2

import google.protobuf.text_format as txtf

import numpy as np
import argparse


class ModelProtoGenerator(object):
    '''
    Class for generating Caffe CNN models through protobuffer files.
    '''
    def __init__(self, initialization='msra', use_cudnn_engine=False):
        # set up the engines
        self.conv_engine = None
        self.spp_engine = None
        if use_cudnn_engine:
            self.conv_engine = P.Convolution.CUDNN
            self.spp_engine = P.SPP.CUDNN
        else:
            self.conv_engine = P.Convolution.CAFFE
            self.spp_engine = P.SPP.CAFFE
        self.phase_train = caffe_pb2.Phase.DESCRIPTOR.values_by_name['TRAIN'].number
        self.phase_test = caffe_pb2.Phase.DESCRIPTOR.values_by_name['TEST'].number
        self.initialization = initialization
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_phocnet_data(self, n, generate_deploy, word_image_lmdb_path, phoc_lmdb_path):
        if generate_deploy:
            n.word_images = L.Input(shape=dict(dim=[1, 1, 100, 250]))
        else:
            n.word_images, n.label = L.Data(batch_size=1, backend=P.Data.LMDB, source=word_image_lmdb_path, prefetch=20,
                                            transform_param=dict(mean_value=255, scale=-1. / 255,), ntop=2)
            n.phocs, n.label_phocs = L.Data(batch_size=1, backend=P.Data.LMDB, source=phoc_lmdb_path, prefetch=20,
                                            ntop=2)

    def set_phocnet_conv_body(self, n, relu_in_place):
        n.conv1_1, n.relu1_1 = self.conv_relu(n.word_images, nout=64, relu_in_place=relu_in_place)
        n.conv1_2, n.relu1_2 = self.conv_relu(n.relu1_1, nout=64, relu_in_place=relu_in_place)
        n.pool1 = L.Pooling(n.relu1_2, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

        n.conv2_1, n.relu2_1 = self.conv_relu(n.pool1, nout=128, relu_in_place=relu_in_place)
        n.conv2_2, n.relu2_2 = self.conv_relu(n.relu2_1, nout=128, relu_in_place=relu_in_place)
        n.pool2 = L.Pooling(n.relu2_2, pooling_param=dict(pool=P.Pooling.MAX, kernel_size=2, stride=2))

        n.conv3_1, n.relu3_1 = self.conv_relu(n.pool2, nout=256, relu_in_place=relu_in_place)
        n.conv3_2, n.relu3_2 = self.conv_relu(n.relu3_1, nout=256, relu_in_place=relu_in_place)
        n.conv3_3, n.relu3_3 = self.conv_relu(n.relu3_2, nout=256, relu_in_place=relu_in_place)
        n.conv3_4, n.relu3_4 = self.conv_relu(n.relu3_3, nout=256, relu_in_place=relu_in_place)
        n.conv3_5, n.relu3_5 = self.conv_relu(n.relu3_4, nout=256, relu_in_place=relu_in_place)
        n.conv3_6, n.relu3_6 = self.conv_relu(n.relu3_5, nout=256, relu_in_place=relu_in_place)

        n.conv4_1, n.relu4_1 = self.conv_relu(n.relu3_6, nout=512, relu_in_place=relu_in_place)
        n.conv4_2, n.relu4_2 = self.conv_relu(n.relu4_1, nout=512, relu_in_place=relu_in_place)
        n.conv4_3, n.relu4_3 = self.conv_relu(n.relu4_2, nout=512, relu_in_place=relu_in_place)
        
        
    


    def conv_relu(self, bottom, nout, kernel_size=3, stride=1, pad=1, relu_in_place=True):
        '''
        Helper method for returning a ReLU activated Conv layer
        '''
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride,
                             num_output=nout, pad=pad, engine=self.conv_engine,
                             weight_filler=dict(type=self.initialization),
                             bias_filler=dict(type='constant'))
        return conv, L.ReLU(conv, in_place=relu_in_place)
    
    def bn_relu_conv(self, bottom, kernel_size, nout, stride, pad, dropout_ratio=0.0):
        '''
        Helper method for returning a ReLU activated Conv layer with batch normalization. It can bes specified also 
        if the layer should make use of Dropout as well.
        '''
        batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        relu = L.ReLU(scale, in_place=True)
        conv = L.Convolution(relu, kernel_size=kernel_size, stride=stride, engine=self.conv_engine, 
                             num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        if dropout_ratio>0:
            drop = L.Dropout(conv, dropout_ratio=dropout_ratio,in_place=True, include=dict(phase=self.phase_train))
        else:
            drop = None
        return batch_norm, scale, relu, conv, drop
    
    
    def relu_conv(self, bottom, kernel_size, nout, stride, pad, dropout_ratio=0.0):
        '''
        Helper method for returning a ReLU activated Conv layer. It can bes specified also 
        if the layer should make use of Dropout as well.
        '''
        
        conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, engine=self.conv_engine, 
                             num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        relu = L.ReLU(conv, in_place=True)
        
        if dropout_ratio>0:
            drop = L.Dropout(conv, dropout_ratio=dropout_ratio,in_place=True, include=dict(phase=self.phase_train))
        else:
            drop = None
            
        return relu, conv, drop
    
    def transition(self, bottom, num_filter, dropout_ratio, no_batch_normalization):
        if no_batch_normalization:
            relu, conv, drop = self.relu_conv(bottom, kernel_size=1, nout=num_filter, stride=1, pad=0, dropout_ratio=dropout_ratio)
        else:
            batch_norm, scale, relu, conv, drop = self.bn_relu_conv(bottom, kernel_size=1, nout=num_filter, stride=1, pad=0, dropout_ratio=dropout_ratio)
            
        pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
        return conv, pooling
    

    def fc_relu(self, bottom, layer_size, dropout_ratio=0.0, relu_in_place=True):
        '''
        Helper method for returning a ReLU activated Fully Connected layer. It can be specified also
        if the layer should make use of Dropout as well.
        '''
        fc = L.InnerProduct(bottom, num_output=layer_size,
                            weight_filler=dict(type=self.initialization),
                            bias_filler=dict(type='constant'))
        relu = L.ReLU(fc, in_place=relu_in_place)
        if dropout_ratio == 0.0:
            return fc, relu
        else:
            return fc, relu, L.Dropout(relu, dropout_ratio=0.5, in_place=True, include=dict(phase=self.phase_train))
        
        
    def set_dense_body(self, n, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, conv_init, init_7):
        '''
        Helper method for the dense body of a DensePHOCNet
        '''
        
        conv_init = np.array(conv_init)
        dropout_ratio = 0.0
        
        if init_7:
            nchannels = 64
            n.conv_init = L.Convolution(n.word_images, kernel_size=7, stride=2, num_output=nchannels,
                                        pad=3, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
                                        engine=self.conv_engine)
            bottom = n.conv_init
            if pool_init:
                if not no_batch_normalization:
                    n.bn_init = L.BatchNorm(bottom, in_place=False, eps = 0.00001, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
                    n.scale_init = L.Scale(n.bn_init, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
                    n.relu_init = L.ReLU(n.scale_init, in_place=True)
                else:
                    n.relu_init = L.ReLU(bottom, in_place=True)
                
                n.pool_init = L.Pooling(n.relu_init, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)
                
                bottom = n.pool_init
                
        else:
            bottom = n.word_images
            

            if conv_init.size > 1:
                for i in range(conv_init.size):    
                    n['Conv_Init%i' % i] = L.Convolution(bottom, kernel_size=3, stride=1, num_output=conv_init[i],
                                                         pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
                                                         engine=self.conv_engine)
                    
                    n.bn_init = L.BatchNorm(n['Conv_Init%i' % i], in_place=False, eps = 0.00001, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
                    n.scale_init = L.Scale(n.bn_init, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
                    n.relu_init = L.ReLU(n.scale_init, in_place=True)
                        
                    bottom = n.relu_init
                    nchannels = conv_init[i]
            else:
                n['Conv_Init0'] = L.Convolution(bottom, kernel_size=3, stride=1, num_output=int(conv_init),
                                                     pad=1, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
                                                     engine=self.conv_engine)
                    
                n.bn_init = L.BatchNorm(n['Conv_Init0'], in_place=False, eps = 0.00001, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
                n.scale_init = L.Scale(n.bn_init, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
                n.relu_init = L.ReLU(n.scale_init, in_place=True)
                 
                bottom = n.relu_init
                nchannels = int(conv_init)
                    

            if pool_init:         
                n.pool_init = L.Pooling(n.relu_init, pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)
                        
            bottom = n.pool_init
                 
        temp = bottom
            
    
        
        s_bn = 0.5 if use_bottleneck else 1
        
        if config is None:
            N = (nlayers-4-(nblocks-1))/nblocks * s_bn * np.ones(nblocks)
        else:
            N = np.array(config)
        
        
        for b in range(nblocks):
            #Build Dense Block b
            for i in range(int(N[b])):
                if not no_batch_normalization:
                    bn = 'conv%i_%i/bn' % (b,i)
                    scale = 'conv%i_%i/scale' % (b,i)
                    
                relu = 'relu%i_%i' % (b,i)
                conv = 'conv%i_%i' % (b,i)
                drop = 'dropout%i_%i' % (b,i)
                concat = 'concat%i_%i' % (b,i)
                
                if use_bottleneck:
                    if not no_batch_normalization:
                        bn = 'conv%i_%i/x1/bn' % (b,i)
                        scale = 'conv%i_%i/x1/scale' % (b,i)
                    relu = 'relu%i_%i/x1' % (b,i)
                    conv = 'conv%i_%i/x1' % (b,i)
                    
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], _ = self.bn_relu_conv(bottom, kernel_size=1, nout=4*growth_rate, 
                                                                                 stride=1, pad=0, dropout_ratio=0)
                    else:
                        n[relu], n[conv], _ = self.relu_conv(bottom, kernel_size=1, nout=4*growth_rate, 
                                                             stride=1, pad=0, dropout_ratio=0)
                        
                    temp = n[conv]
                    
                    
                    if not no_batch_normalization:
                        bn = 'conv%i_%i/x2/bn' % (b,i)
                        scale = 'conv%i_%i/x2/scale' % (b,i)
                        
                    relu = 'relu%i_%i/x2' % (b,i)
                    conv = 'conv%i_%i/x2' % (b,i)
                        
                if dropout_ratio > 0:
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], n[drop] = self.bn_relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                                       stride=1, pad=1, dropout_ratio=dropout_ratio)
                    else:
                        n[relu], n[conv], n[drop] = self.relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                   stride=1, pad=1, dropout_ratio=dropout_ratio)
                        
                    n[concat] = L.Concat(bottom, n[drop], axis=1)
                else:
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], _ = self.bn_relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                                 stride=1, pad=1, dropout_ratio=dropout_ratio)
                    else:
                        n[relu], n[conv], _ = self.relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                stride=1, pad=1, dropout_ratio=dropout_ratio)
                        
                    n[concat] = L.Concat(bottom, n[conv], axis=1)
                    
                bottom = n[concat]
                temp = n[concat]
                
                nchannels += growth_rate
              
            if b < nblocks-1:
                C = 0.5 if use_compression else 1
                nchannels = nchannels*C
                n['conv%i_blk' % b],n['pool%i' % b] = self.transition(n[concat], int(nchannels), dropout_ratio, no_batch_normalization)
                bottom = n['pool%i' % b]
                temp = bottom
                
                  
        return bottom, nchannels
        


    
    def get_phocnet(self, word_image_lmdb_path, phoc_lmdb_path, phoc_size, pooling, max_out, use_perm, tpp_levels=5, 
                        generate_deploy=False):
        '''
        Returns a NetSpec definition of the TPP-PHOCNet with maxout pruning. The definition can then be transformed
        into a protobuffer message by casting it into a str.
        '''

        
        n = NetSpec()
        # Data
        self.set_phocnet_data(n=n, generate_deploy=generate_deploy,
                              word_image_lmdb_path=word_image_lmdb_path,
                              phoc_lmdb_path=phoc_lmdb_path)

        # Conv Part
        self.set_phocnet_conv_body(n=n, relu_in_place=True)
        
        
        if pooling == 'tpp':
            n.tpp5 = L.TPP(n.relu4_3, tpp_param=dict(pool=P.TPP.MAX, pyramid_layer=range(1, tpp_levels + 1), engine=self.spp_engine))
            bottom = n.tpp5
            size_tpp = 7680
        elif pooling == 'ave':
            n.gp = L.Pooling(n.relu4_3, pool=P.Pooling.AVE, global_pooling=True )
            bottom = n.gp
            size_tpp = 512
        elif pooling == 'max':
            n.gp = L.Pooling(n.relu4_3, pool=P.Pooling.MAX, global_pooling=True )
            bottom = n.gp
            size_tpp = 512
        elif pooling == 'spp':
            n.spp5 = L.SPP(n.relu4_3, spp_param=dict(pool=P.SPP.MAX, pyramid_height=3, engine=self.spp_engine))
            bottom = n.spp5
            size_tpp = 10752
        else:
            raise ValueError('Pooling layer %s unknown' % self.pooling)
        
            
        # Maxout Pruning
        k = max_out
        
        if max_out > 0: 
            n.reshape = L.Reshape(bottom, shape=dict(dim=[1, size_tpp/k,k,1]))
                
            n.premax = L.Scale(n.reshape, bias_term=False, in_place=False, filler=dict(value=1), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = 3)
            n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
            
            n.fc6, n.relu6, n.drop6 = self.fc_relu(bottom=n.maxout, layer_size=4096,
                                                   dropout_ratio=0.5, relu_in_place=True)      
        else:
            n.fc6, n.relu6, n.drop6 = self.fc_relu(bottom=bottom, layer_size=4096,
                                                   dropout_ratio=0.5, relu_in_place=True)
        
        
        # FC Part
        n.fc7, n.relu7, n.drop7 = self.fc_relu(bottom=n.drop6, layer_size=4096,
                                               dropout_ratio=0.5, relu_in_place=True)
        n.fc8 = L.InnerProduct(n.drop7, num_output=phoc_size,
                               weight_filler=dict(type=self.initialization),
                               bias_filler=dict(type='constant'))
        n.sigmoid = L.Sigmoid(n.fc8, include=dict(phase=self.phase_test))

        # output part
        if not generate_deploy:
            n.silence = L.Silence(n.sigmoid, ntop=0, include=dict(phase=self.phase_test))
            n.loss = L.SigmoidCrossEntropyLoss(n.fc8, n.phocs)

        return n.to_proto()

    
    def get_dense_phocnet(self, word_image_lmdb_path, phoc_lmdb_path, phoc_size, pooling, 
                          nblocks, growth_rate, nlayers, config, pool_init, conv_init, init_7,
                          no_batch_normalization, use_bottleneck, use_compression, max_out,
                          generate_deploy=False):
        
        n = NetSpec()
        # Data
        self.set_phocnet_data(n=n, generate_deploy=generate_deploy,
                              word_image_lmdb_path=word_image_lmdb_path,
                              phoc_lmdb_path=phoc_lmdb_path)

        end, fm_depth = self.set_dense_body(n, nblocks=nblocks, growth_rate=growth_rate, nlayers=nlayers, config=config,
                                            no_batch_normalization=no_batch_normalization, use_bottleneck=use_bottleneck,
                                            use_compression=use_compression, pool_init=pool_init, init_7=init_7, conv_init=conv_init)
        
        
        
        
        # Pooling layer
        tpp_levels = 5
        
        if pooling == 'tpp':
            n.tpp5 = L.TPP(end, tpp_param=dict(pool=P.TPP.MAX, pyramid_layer=range(1, tpp_levels + 1), engine=self.spp_engine))
            bottom = n.tpp5
            pooling_factor = 15
        elif pooling == 'ave':
            n.gp = L.Pooling(end, pool=P.Pooling.AVE, global_pooling=True )
            bottom = n.gp
            pooling_factor = 1
        elif pooling == 'max':
            n.gp = L.Pooling(end, pool=P.Pooling.MAX, global_pooling=True )
            bottom = n.gp
            pooling_factor = 1
        elif pooling == 'spp':
            n.spp5 = L.SPP(end, spp_param=dict(pool=P.SPP.MAX, pyramid_height=3, engine=self.spp_engine))
            bottom = n.spp5
            pooling_factor = 21
        else:
            raise ValueError('Pooling layer %s unknown' % self.pooling)
            

                   
        # Maxout
        k = max_out
        
        if max_out >= 0: 
            k = max_out
            size_pp = fm_depth*pooling_factor
       
            n.reshape = L.Reshape(bottom, shape=dict(dim=[1, size_pp/k,k,1]))
                
            n.premax = L.Scale(n.reshape, bias_term=False, in_place=False, filler=dict(value=1), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = 3)
            n.maxout = L.Pooling(n.premax, pool=P.Pooling.MAX, global_pooling=True)
            
            n.fc6_d, n.relu6, n.drop6 = self.fc_relu(bottom=n.maxout, layer_size=4096,
                                                   dropout_ratio=0.5, relu_in_place=True)      
        else:
            n.fc6_d, n.relu6, n.drop6 = self.fc_relu(bottom=bottom, layer_size=4096,
                                                   dropout_ratio=0.5, relu_in_place=True)
            
        
        # FC Part
        n.fc7_d, n.relu7, n.drop7 = self.fc_relu(bottom=n.drop6, layer_size=4096,
                                                 dropout_ratio=0.5, relu_in_place=True)
        n.fc8_d = L.InnerProduct(n.drop7, num_output=phoc_size,
                                 weight_filler=dict(type=self.initialization),
                                 bias_filler=dict(type='constant'))
        n.sigmoid = L.Sigmoid(n.fc8_d, include=dict(phase=self.phase_test))
            
            

        # output part
        if not generate_deploy:
            n.silence = L.Silence(n.sigmoid, ntop=0, include=dict(phase=self.phase_test))
            n.loss = L.SigmoidCrossEntropyLoss(n.fc8_d, n.phocs)
            
        return n.to_proto()
    

def main():
    '''
    this module can be called as main function in which case it prints the
    prototxt definition for the given net
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_architecture', '-ca', choices=['phocnet', 'tpp-phocnet'], default='phocnet',
                        help='The CNN architecture to print to standard out')
    parser.add_argument('--word_image_lmdb_path', '-wilp', default='./word_images_lmdb')
    parser.add_argument('--phoc_lmdb_path', '-plp', default='./phoc_lmdb')
    parser.add_argument('--phoc_size', type=int, default=604)
    args = parser.parse_args()
    if args.cnn_architecture == 'phocnet':
        print str(ModelProtoGenerator().get_phocnet(word_image_lmdb_path=args.word_image_lmdb_path,
                                                    phoc_lmdb_path=args.phoc_lmdb_path,
                                                    phoc_size=args.phoc_size,
                                                    generate_deploy=args.generate_deploy))
    elif args.cnn_architecture == 'tpp-phocnet':
        print str(ModelProtoGenerator().get_tpp_phocnet(word_image_lmdb_path=args.word_image_lmdb_path,
                                                        phoc_lmdb_path=args.phoc_lmdb_path,
                                                        phoc_size=args.phoc_size,
                                                        generate_deploy=args.generate_deploy))

if __name__ == '__main__':
    main()

