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
        
        
    def set_phocnet_dense_body(self, n, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, conv_init, init_7, dropout_ratio):
        conv_init = np.array(conv_init)
        
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
        

    def set_phocnet_dense_body_from_file(self, n, dense_net_file):
        net = caffe_pb2.NetParameter()
        
        with open(dense_net_file) as f:
            s = f.read()
            txtf.Merge(s, net)

        # Convolutional Layer for grey scale images
        n['conv1_grey'] = L.Convolution(n.word_images, kernel_size=7, stride=2, bias_term = False,
                             num_output=64, pad=3,
                             weight_filler=dict(type=self.initialization),
                             bias_filler=dict(type='constant'),
                             engine=self.conv_engine)
        
        n[net.layer[1].name] = L.BatchNorm(n['conv1_grey'], in_place=False, eps = 0.00001)
        
        # Process Base file
        for l in net.layer:
            if unicode(l.bottom).split('\'')[1] != 'data' and unicode(l.bottom).split('\'')[1] != 'conv1':
                n[l.name] = self.add_layer(n, l)
                end = l.name
                
        return n[l.name]
    
    
    def get_dense_triplet_net_from_file(self, n, dense_net_file):
        n.data = L.Input(shape=dict(dim=[1, 3, 100, 250]))
        
        net = caffe_pb2.NetParameter()
        
        with open(dense_net_file) as f:
            s = f.read()
            txtf.Merge(s, net)

        
        # Process Base file
        for l in net.layer:
            n[l.name] = self.add_layer(n, l)
            end = l.name
                
        return n[l.name]

                
    def add_layer(self, net, layer):
        bottom_str = unicode(layer.bottom).split('\'')[1]
        
        if layer.type == 'Convolution':
            return L.Convolution(net[bottom_str],
                                 name = layer.name,
                                 num_output = layer.convolution_param.num_output,
                                 bias_term = layer.convolution_param.bias_term,
                                 pad = [0 if int(filter(str.isdigit, str(layer.convolution_param.kernel_size))) == 1 else int(filter(str.isdigit, str(layer.convolution_param.pad)))],
                                 kernel_size = int(filter(str.isdigit, str(layer.convolution_param.kernel_size))),
                                 stride = [1 if not str(layer.convolution_param.stride)[1].isdigit() else int(filter(str.isdigit, str(layer.convolution_param.stride)))])
            
        elif layer.type == 'Pooling':
            return L.Pooling(net[bottom_str],
                             name = layer.name,
                             pool = layer.pooling_param.pool,
                             kernel_size = layer.pooling_param.kernel_size,
                             stride = layer.pooling_param.stride)
        elif layer.type == 'BatchNorm':
            return L.BatchNorm(net[bottom_str],
                               name = layer.name, 
                               in_place=False, 
                               eps = layer.batch_norm_param.eps)
        elif layer.type == 'Concat':
            return L.Concat(net[layer.bottom[0]],
                            net[layer.bottom[1]],
                            name = layer.name)
        elif layer.type == 'ReLU':
            return L.ReLU(net[bottom_str],
                          name = layer.name,
                          in_place = False)
        elif layer.type == 'Scale':
            return L.Scale(net[bottom_str],
                           name = layer.name,
                           bias_term = layer.scale_param.bias_term,
                           in_place = False)
           


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
    
    def bn_relu_conv_tri(self, bottom, kernel_size, nout, stride, pad, param_name, dropout_ratio=0.0):
        '''
        Helper method for returning a ReLU activated Conv layer with batch normalization. It can bes specified also 
        if the layer should make use of Dropout as well.
        '''
        batch_norm = L.BatchNorm(bottom, in_place=False, eps=0.00001,param=[dict(name='conv' + param_name + '/bn' + '_w1'), dict(name='conv' + param_name + '/bn' + '_w2'), dict(name='conv' + param_name + '/bn' + '_w3')])
        scale = L.Scale(batch_norm, bias_term=True, in_place=True, param=[dict(name='conv' + param_name + '/scale' + '_w'), dict(name='conv' + param_name + '/scale' + '_b')])
        relu = L.ReLU(scale, in_place=True)
        
        if pad > 0:
            conv = L.Convolution(relu, kernel_size=kernel_size, num_output=nout, pad = pad, bias_term=False, param=[dict(name='conv' + param_name +'_w')])
        else:
            conv = L.Convolution(relu, kernel_size=kernel_size, num_output=nout, bias_term=False, param=[dict(name='conv' + param_name +'_w')])
            
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
    
    def transition_tri(self, bottom, num_filter, dropout_ratio, no_batch_normalization, param_name):
        if no_batch_normalization:
            relu, conv, drop = self.relu_conv(bottom, kernel_size=1, nout=num_filter, stride=1, dropout_ratio=dropout_ratio)
        else:
            batch_norm, scale, relu, conv, drop = self.bn_relu_conv_tri(bottom, kernel_size=1, nout=num_filter, stride=1, pad=0, param_name=param_name, dropout_ratio=dropout_ratio)
            
        pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
        return batch_norm, scale, relu, conv, pooling

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


    def get_tpp_phocnet(self, word_image_lmdb_path, phoc_lmdb_path, phoc_size, tpp_levels=5,
                        generate_deploy=False):
        '''
        Returns a NetSpec definition of the TPP-PHOCNet. The definition can then be transformed
        into a protobuffer message by casting it into a str.
        '''
        
        n = NetSpec()
        # Data
        self.set_phocnet_data(n=n, generate_deploy=generate_deploy,
                              word_image_lmdb_path=word_image_lmdb_path,
                              phoc_lmdb_path=phoc_lmdb_path)

        # Conv Part
        self.set_phocnet_conv_body(n=n, relu_in_place=True)

        # FC Part
        n.tpp5 = L.TPP(n.relu4_3, tpp_param=dict(pool=P.TPP.MAX, pyramid_layer=range(1, tpp_levels + 1), engine=self.spp_engine))
        n.fc6, n.relu6, n.drop6 = self.fc_relu(bottom=n.tpp5, layer_size=4096,
                                               dropout_ratio=0.5, relu_in_place=True)
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
            fm_size = 15
            bottom = n.tpp5
            size_tpp = 7680
        elif pooling == 'ave':
            n.gp = L.Pooling(n.relu4_3, pool=P.Pooling.AVE, global_pooling=True )
            fm_size = 1
            bottom = n.gp
            size_tpp = 512
        elif pooling == 'max':
            n.gp = L.Pooling(n.relu4_3, pool=P.Pooling.MAX, global_pooling=True )
            fm_size = 1
            bottom = n.gp
            size_tpp = 512
        elif pooling == 'spp':
            n.spp5 = L.SPP(n.relu4_3, spp_param=dict(pool=P.SPP.MAX, pyramid_height=3, engine=self.spp_engine))
            fm_size = 21
            bottom = n.spp5
            size_tpp = 10752
        else:
            raise ValueError('Pooling layer %s unknown' % self.pooling)
        
        
        if use_perm:
            n.perm = L.Python(bottom, module='permute', layer='PermuteLayer', param_str = '{ "fm_size": %i, "k": %i }' % (fm_size, max_out))
            bottom = n.perm
            
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

    
    def get_dense_phocnet(self, word_image_lmdb_path, phoc_lmdb_path, phoc_size, dense_net_file,
                        pooling, nblocks, growth_rate, nlayers, config, pool_init, conv_init, init_7,
                        no_batch_normalization,
                        use_bottleneck, use_compression, dropout_ratio, max_out, use_perm,
                        generate_deploy=False):
        
        n = NetSpec()
        # Data
        self.set_phocnet_data(n=n, generate_deploy=generate_deploy,
                              word_image_lmdb_path=word_image_lmdb_path,
                              phoc_lmdb_path=phoc_lmdb_path)
        
        # Dense Body from File
        if dense_net_file is not None:
            end = self.set_phocnet_dense_body_from_file(n, dense_net_file)
        else:
            end, fm_depth = self.set_phocnet_dense_body(n, nblocks=nblocks, growth_rate=growth_rate, nlayers=nlayers, config=config, 
                                                        no_batch_normalization=no_batch_normalization, use_bottleneck=use_bottleneck, 
                                                        use_compression=use_compression, pool_init=pool_init, init_7=init_7, conv_init=conv_init,
                                                        dropout_ratio=dropout_ratio)
        
        
        
        
        # Pooling layer
        tpp_levels = 5
        
        if pooling == 'tpp':
            n.tpp5 = L.TPP(end, tpp_param=dict(pool=P.TPP.MAX, pyramid_layer=range(1, tpp_levels + 1), engine=self.spp_engine))
            fm_size = 15
            bottom = n.tpp5
        elif pooling == 'ave':
            n.gp = L.Pooling(end, pool=P.Pooling.AVE, global_pooling=True )
            fm_size = 1
            bottom = n.gp
        elif pooling == 'max':
            n.gp = L.Pooling(end, pool=P.Pooling.MAX, global_pooling=True )
            fm_size = 1
            bottom = n.gp
        elif pooling == 'spp':
            n.spp5 = L.SPP(end, spp_param=dict(pool=P.SPP.MAX, pyramid_height=3, engine=self.spp_engine))
            fm_size = 21
            bottom = n.spp5
        else:
            raise ValueError('Pooling layer %s unknown' % self.pooling)
            
        # Permutation layer
        if use_perm:
            n.perm = L.Python(bottom, module='permute', layer='PermuteLayer', param_str = '{ "fm_size": %i, "k": %i }' % (fm_size, max_out))
            bottom = n.perm
            if fm_size == 1:
                self.logger.info('Permutation Layer for Global Pooling applied. fm_size = 1. Check if valid!')
                   
        # Maxout Pruning
        k = max_out
        
        if max_out >= 0: 
            k = max_out
            size_pp = fm_depth*15
       
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
    
    
    def get_triplet_net(self, train_word_images_lmdb_path, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio, descriptor_size=1000):
        n = NetSpec()
        
        n.p1, n.p2, n.n = L.Python(ntop = 3, name = 'TripletInput', module='triplet', layer='TripletLayer', param_str = '{ "source": "%s"}' % (train_word_images_lmdb_path))
        
        fc_p1 = self.get_dense_string(n, n.p1, '_p1', nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio, descriptor_size=1000)
        fc_p2 = self.get_dense_string(n, n.p2, '_p2', nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio, descriptor_size=1000)
        fc_n = self.get_dense_string(n, n.n, '_n', nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio, descriptor_size=1000)
        
        
        #Output
        n.p1_n = L.EuclideanSimilarity(fc_p1,fc_n)
        
        n.p2_n = L.EuclideanSimilarity(fc_p2,fc_n)
        
        n.p1_p2 = L.EuclideanSimilarity(fc_p1,fc_p2)
        
        
        # Find minimal negative distance
        n.argmax = L.Eltwise(n.p1_n, n.p2_n, operation = P.Eltwise.MAX)
        
        n.scale_neg = L.Scale(n.argmax, bias_term=False, in_place=False, filler=dict(value=-2), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        n.neg = L.Power(n.scale_neg, power = 0.5)
        
        
        
        n.scale_pos =  L.Scale(n.p1_p2, bias_term=False, in_place=False, filler=dict(value=-2), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        n.pos = L.Power(n.scale_pos, power = 0.5)
        
        n.concat = L.Concat(n.pos, n.neg, axis=3)
        
        n.sm = L.Softmax(n.concat)
        
        
        n.target0 = L.DummyData(num=1,channels=1,height = 1,width=1,data_filler=dict(value=0))
        n.target1 = L.DummyData(num=1,channels=1,height = 1,width=1,data_filler=dict(value=1))
        
        n.target = L.Concat(n.target0, n.target1, axis=3)
        
        n.euclidean_loss = L.EuclideanLoss(n.sm,n.target)
        
        return n.to_proto()

    
    
    def get_dense_string(self, n, data, string, nblocks, growth_rate, nlayers, config, no_batch_normalization, use_bottleneck, use_compression, pool_init, dropout_ratio, descriptor_size=1000):
        nchannels = 64
        
        n['conv1' + string] = L.Convolution(data, kernel_size=7, stride=2, num_output=nchannels,pad=3, bias_term=False,
                                            param=[dict(name='conv1_w')])
            
        bottom = n['conv1' + string]
        temp = n['conv1' + string]
            
        if pool_init:
            if not no_batch_normalization:
                n['conv1/bn' + string] = L.BatchNorm(n['conv1' + string], in_place=False, eps = 0.00001,
                                                     param=[dict(name='conv1/bn_w1'), dict(name='conv1/bn_w2'), dict(name='conv1/bn_w3')])
                n['conv1/scale' + string] = L.Scale(n['conv1/bn' + string], bias_term=True, in_place=True,
                                                    param=[dict(name='conv1/scale_w'), dict(name='conv1/scale_b')])
                n['relu1' + string] = L.ReLU(n['conv1/scale' + string], in_place=True)
            else:
                n['relu1' + string] = L.ReLU(n['conv1/scale' + string], in_place=True)
            
            n['pool1' + string] = L.Pooling(n['relu1' + string], pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)
            
            bottom = n['pool1' + string]
            temp = n['pool1' + string]
        
        
        
        s_bn = 0.5 if use_bottleneck else 1
        
        if config is None:
            N = (nlayers-4-(nblocks-1))/nblocks * s_bn * np.ones(nblocks)
        else:
            N = np.array(config)
        
        
        for b in range(nblocks):
            #Build Dense Block b
            for i in range(int(N[b])):
                if not no_batch_normalization:
                    bn = ('conv%i_%i/bn' % (b+2,i+1))  + string
                    scale = ('conv%i_%i/scale' % (b+2,i+1)) + string
                    
                relu = ('relu%i_%i' % (b+2,i+1)) + string
                conv = ('conv%i_%i' % (b+2,i+1)) + string
                drop = ('dropout%i_%i' % (b+2,i+1)) + string
                concat = ('concat_%i_%i' % (b+2,i+1)) + string
                
                if use_bottleneck:
                    if not no_batch_normalization:
                        bn = ('conv%i_%i/x1/bn' % (b+2,i+1)) + string
                        scale = ('conv%i_%i/x1/scale' % (b+2,i+1)) + string
                    relu = ('relu%i_%i/x1' % (b+2,i+1)) + string
                    conv = ('conv%i_%i/x1' % (b+2,i+1)) + string
                    
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], _ = self.bn_relu_conv_tri(bottom, kernel_size=1, nout=4*growth_rate, 
                                                                                 stride=1, pad=0, param_name = ('%i_%i/x1' % (b+2,i+1)), dropout_ratio=0)
                    else:
                        n[relu], n[conv], _ = self.relu_conv(bottom, kernel_size=1, nout=4*growth_rate, 
                                                             stride=1, pad=0, param_name = ('%i_%i/x1' % (b+2,i+1)), dropout_ratio=0)
                        
                    temp = n[conv]
                    
                    
                    if not no_batch_normalization:
                        bn = ('conv%i_%i/x2/bn' % (b+2,i+1)) + string
                        scale = ('conv%i_%i/x2/scale' % (b+2,i+1)) + string
                        
                    relu = ('relu%i_%i/x2' % (b+2,i+1)) + string
                    conv = ('conv%i_%i/x2' % (b+2,i+1)) + string
                        
                if dropout_ratio > 0:
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], n[drop] = self.bn_relu_conv_tri(temp, kernel_size=3, nout=growth_rate, 
                                                                                       stride=1, pad=1, param_name = ('%i_%i/x2' % (b+2,i+1)), dropout_ratio=dropout_ratio)
                    else:
                        n[relu], n[conv], n[drop] = self.relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                   stride=1, pad=1, param_name = ('%i_%i/x2' % (b+2,i+1)), dropout_ratio=dropout_ratio)
                        
                    n[concat] = L.Concat(bottom, n[drop])
                else:
                    if not no_batch_normalization:
                        n[bn], n[scale], n[relu], n[conv], _ = self.bn_relu_conv_tri(temp, kernel_size=3, nout=growth_rate, 
                                                                                 stride=1, pad=1, param_name = ('%i_%i/x2' % (b+2,i+1)), dropout_ratio=dropout_ratio)
                    else:
                        n[relu], n[conv], _ = self.relu_conv(temp, kernel_size=3, nout=growth_rate, 
                                                                stride=1, pad=1, param_name = ('%i_%i/x2' % (b+2,i+1)), dropout_ratio=dropout_ratio)
                        
                    n[concat] = L.Concat(bottom, n[conv])
                    
                bottom = n[concat]
                temp = n[concat]
                
                nchannels += growth_rate
              
            if b < nblocks-1:
                C = 0.5 if use_compression else 1
                nchannels = nchannels*C
                n[('conv%i_blk/bn' % (b+2)) + string],n[('conv%i_blk/scale' % (b+2)) + string],n[('relu%i_blk' % (b+2)) + string],n[('conv%i_blk' % (b+2)) + string],n[('pool%i' % (b+2)) + string] = self.transition_tri(n[concat], int(nchannels), dropout_ratio, no_batch_normalization, param_name = ('%i_blk' % (b+2)))
                bottom = n[('pool%i' % (b+2)) + string]
                temp = bottom
        
        n[('conv%i_blk/bn' % (nblocks+1)) + string] = L.BatchNorm(temp, in_place=False, eps=0.00001,
                                                                  param=[dict(name=('conv%i_blk/bn_w1'% (nblocks+1))), dict(name=('conv%i_blk/bn_w2'% (nblocks+1))), dict(name=('conv%i_blk/bn_w3'% (nblocks+1)))])
        n[('conv%i_blk/scale' % (nblocks+1)) + string] = L.Scale(n[('conv%i_blk/bn' % (nblocks+1)) + string], bias_term=True, in_place=True,
                                                                 param=[dict(name=('conv%i_blk/scale_w' % (nblocks+1))), dict(name=('conv%i_blk/scale_b' % (nblocks+1)))])
        n[('relu%i_blk' % (nblocks+1)) + string] = L.ReLU(n[('conv%i_blk/scale' % (nblocks+1)) + string], in_place=True)
        
        
        n[('pool%i' % (nblocks+1)) + string] = L.Pooling(n[('relu%i_blk' % (nblocks+1)) + string], pool=P.Pooling.AVE, global_pooling=True)
        
        n[('fc6') + string] = L.Convolution(n[('pool%i' % (nblocks+1)) + string], kernel_size=1, num_output=descriptor_size,
                                            param=[dict(name='fc6_w'), dict(name='fc6_b')])
        
    
        
        return n[('fc6') + string]
    
    def get_triplet_net_121(self, word_image_lmdb_path):
        
        n = NetSpec()
        
        '''
        # Input
        n.p1, n.p2, n.n = L.Python(ntop = 3, name = 'TripletInput', module='triplet', layer='TripletLayer', param_str = '{ "source": "%s"}' % (word_image_lmdb_path))
        
        
        # Dummy 1
        n.batch_norm_1 = L.BatchNorm(n.p1, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        n.scale_1 = L.Scale(n.batch_norm_1, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        n.relu_1 = L.ReLU(n.scale_1, in_place=True)
        n.conv_1 = L.Convolution(n.relu_1, kernel_size=3, stride=1, engine=self.conv_engine, 
                             num_output=32, pad=2, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.pool_1 =  L.Pooling(n.conv_1, pool=P.Pooling.AVE, global_pooling=True )
        
        n.fc_1 = L.InnerProduct(n.pool_1, num_output=512, weight_filler=dict(type=self.initialization), bias_filler=dict(type='constant'))
        
        # Dummy 2
        n.batch_norm_2 = L.BatchNorm(n.p2, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        n.scale_2 = L.Scale(n.batch_norm_2, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        n.relu_2 = L.ReLU(n.scale_2, in_place=True)
        n.conv_2 = L.Convolution(n.relu_2, kernel_size=3, stride=1, engine=self.conv_engine, 
                             num_output=32, pad=2, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.pool_2 =  L.Pooling(n.conv_2, pool=P.Pooling.AVE, global_pooling=True )
        
        n.fc_2 = L.InnerProduct(n.pool_2, num_output=512, weight_filler=dict(type=self.initialization), bias_filler=dict(type='constant'))
        
        # Dummy 3
        n.batch_norm_3 = L.BatchNorm(n.n, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        n.scale_3 = L.Scale(n.batch_norm_3, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
        n.relu_3 = L.ReLU(n.scale_3, in_place=True)
        n.conv_3 = L.Convolution(n.relu_3, kernel_size=3, stride=1, engine=self.conv_engine, 
                             num_output=32, pad=2, bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
        n.pool_3 =  L.Pooling(n.conv_3, pool=P.Pooling.AVE, global_pooling=True )
        
        n.fc_3 = L.InnerProduct(n.pool_3, num_output=512, weight_filler=dict(type=self.initialization), bias_filler=dict(type='constant'))
        '''
        
        n.p1 = L.Input(shape=dict(dim=[1,3]))
        n.p2 = L.Input(shape=dict(dim=[1,3]))
        n.n = L.Input(shape=dict(dim=[1,3]))
        
        
        #Output
        n.p1_n = L.EuclideanSimilarity(n.p1,n.n)
        
        n.p2_n = L.EuclideanSimilarity(n.p2,n.n)
        
        n.p1_p2 = L.EuclideanSimilarity(n.p1,n.p2)
        
        
        # Find minimal negative distance
        n.argmax = L.Eltwise(n.p1_n, n.p2_n, operation = P.Eltwise.MAX)
        
        n.scale_neg = L.Scale(n.argmax, bias_term=False, in_place=False, filler=dict(value=-2), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        n.neg = L.Power(n.scale_neg, power = 0.5)
        
        
        
        n.scale_pos =  L.Scale(n.p1_p2, bias_term=False, in_place=False, filler=dict(value=-2), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        n.pos = L.Power(n.scale_pos, power = 0.5)
        
        n.concat = L.Concat(n.pos, n.neg, axis=3)
        
        n.sm = L.Softmax(n.concat)
        
        
        n.target0 = L.DummyData(num=1,channels=1,height = 1,width=1,data_filler=dict(value=0))
        n.target1 = L.DummyData(num=1,channels=1,height = 1,width=1,data_filler=dict(value=1))
        
        n.target = L.Concat(n.target0, n.target1, axis=3)
        
        n.euclidean_loss = L.EuclideanLoss(n.sm,n.target)
        
        
        
        return n.to_proto()
    
    def eltwise_min(self, bottom1, bottom2):
        premin1 = L.Scale(bottom1, bias_term=False, in_place=False, filler=dict(value=-1), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        
        premin2 = L.Scale(bottom2, bias_term=False, in_place=False, filler=dict(value=-1), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)

        argmax = L.Eltwise(premin1, premin2, operation = P.Eltwise.MAX)
        
        out = L.Scale(argmax, bias_term=False, in_place=False, filler=dict(value=-1), bias_filler=dict(value=0),
                               param=[dict(lr_mult=0, decay_mult=0)], num_axes = -1)
        
        return out
        
        
        
        
    # deprecated
    '''
    def get_triplet_net(self, single_string, word_image_lmdb_path):
        proto = 'layer {\n  type: "Python"\n  name: "TripletInput"\n  top: "p1"\n  top: "p2"\n  top: "n"\n  python_param {\n    module: "triplet"\n    layer: "TripletLayer"\n    param_str: \'{"source": "%s"}\'\n  }\n}\n' % word_image_lmdb_path
        
        with open(single_string) as f:
            lines = f.readlines()
            
            
            for string in ('p1','p2','n'):
                for k,l in enumerate(lines):
                    if l == 'layer {\n':
                        # get type
                        type = lines[k+2][9:-2]
                        
                        
                        # adapt convolutional layer
                        
                        
                        if type == 'Convolution':
                            c_layer = lines[k:k+20]
                        
                            # bottom
                            if c_layer[1][9:-2] == 'conv_init':
                                c_layer[3] = c_layer[3][:9] + ' "%s"\n' % string
                            else:     
                                c_layer[3] = c_layer[3][:-2] + '_%s' % string + c_layer[3][-2:]
                            
                            # top
                            c_layer[4] = c_layer[4][:-2] + '_%s' % string + c_layer[4][-2:]
                            
                            # name param
                            param_name = c_layer[1][9:-2] + '_w'
                            param = '  param {\n    name: "%s"\w    lr_mult: 1\n  }'
                            
                            c_layer.insert(5, '  param {\n')
                            c_layer.insert(6, '    name: "%s"\n' % param_name)
                            c_layer.insert(7, '    lr_mult: 0\n')
                            c_layer.insert(8, '  }\n')
                            
                            c_layer[1] = c_layer[1][:-2] + ('_%s' % string) + c_layer[1][-2:]
                            for c_line in c_layer:
                                proto = proto + c_line
                                
                        
                        # adapt batch norm layer
                        if type == 'BatchNorm':
                            if lines[k+18] == 'layer {\n':
                                bn_layer = lines[k:k+18]
                            if lines[k+21] == 'layer {\n':
                                bn_layer = lines[k:k+21]
                                
                            # bottom
                            bn_layer[3] = bn_layer[3][:-2] + '_%s' % string + bn_layer[3][-2:]
                            
                            # top
                            bn_layer[4] = bn_layer[4][:-2] + '_%s' % string + bn_layer[4][-2:]
                                
                            param_name = bn_layer[1][9:-2] + '_w'
                            
                            
                            bn_layer.insert(6, '    name: "%s%i"\n' % (param_name, 1))
                            bn_layer.insert(11, '    name: "%s%i"\n' % (param_name, 2))
                            bn_layer.insert(16, '    name: "%s%i"\n' % (param_name, 3))
                            
                            bn_layer[1] = bn_layer[1][:-2] + ('_%s' % string) + bn_layer[1][-2:]
                            for bn_line in bn_layer:
                                proto = proto + bn_line
                        
                           
                        # adapt scale layer
                        if type == 'Scale':
                            s_layer = lines[k:k+15]
                            
                            
                            # bottom
                            s_layer[3] = s_layer[3][:-2] + '_%s' % string + s_layer[3][-2:]
                            
                            # top
                            s_layer[4] = s_layer[4][:-2] + '_%s' % string + s_layer[4][-2:]
                                
                            param_name = s_layer[1][9:-2]
                            
                            s_layer.insert(5, '  param {\n')
                            s_layer.insert(6, '    name: "%s"\n' % (param_name + '_w'))
                            s_layer.insert(7, '    lr_mult: 0\n')
                            s_layer.insert(8, '  }\n')
                            
                            s_layer.insert(9, '  param {\n')
                            s_layer.insert(10, '    name: "%s"\n' % (param_name + '_b'))
                            s_layer.insert(11, '    lr_mult: 0\n')
                            s_layer.insert(12, '  }\n')
                            
                            s_layer[1] = s_layer[1][:-2] + ('_%s' % string) + s_layer[1][-2:]
                            for s_line in s_layer:
                                proto = proto + s_line
                        
                        
                        # adapt relu layer
                        if type == 'ReLU':
                            r_layer = lines[k:k+6]
                            
                            # bottom
                            r_layer[3] = r_layer[3][:-2] + '_%s' % string + r_layer[3][-2:]
                            
                            # top
                            r_layer[4] = r_layer[4][:-2] + '_%s' % string + r_layer[4][-2:]
                            
                            r_layer[1] = r_layer[1][:-2] + ('_%s' % string) + r_layer[1][-2:]
                            for r_line in r_layer:
                                proto = proto + r_line
                                
                        # copy concat layer
                        if type == 'Concat':
                            con_layer = lines[k:k+10]
                                 
                            # bottom 1
                            con_layer[3] = con_layer[3][:-2] + '_%s' % string + con_layer[3][-2:]
                            
                            # bottom 2
                            con_layer[4] = con_layer[4][:-2] + '_%s' % string + con_layer[4][-2:]
                            
                            # top
                            con_layer[5] = con_layer[5][:-2] + '_%s' % string + con_layer[5][-2:]
        
                            con_layer[1] = con_layer[1][:-2] + ('_%s' % string) + con_layer[1][-2:]
                            for con_line in con_layer:
                                proto = proto + con_line
                                
                        # copy pooling layer
                        if type == 'Pooling':
                            
                            
                            if lines[k+12] == 'layer {\n':
                                pool_layer = lines[k:k+12]
                            elif lines[k+11] == 'layer {\n':
                                pool_layer = lines[k:k+11]
                                
                            # bottom
                            pool_layer[3] = pool_layer[3][:-2] + '_%s' % string + pool_layer[3][-2:]
                            
                            # top
                            pool_layer[4] = pool_layer[4][:-2] + '_%s' % string + pool_layer[4][-2:]
                            
                            
                            pool_layer[1] = pool_layer[1][:-2] + ('_%s' % string) + pool_layer[1][-2:]
                            for pool_line in pool_layer:
                                proto = proto + pool_line
                        
                        if type == 'TPP':
                            tpp_layer = lines[k:k+15]
                            
                            # bottom
                            tpp_layer[3] = tpp_layer[3][:-2] + '_%s' % string + tpp_layer[3][-2:]
                            
                            # top
                            tpp_layer[4] = tpp_layer[4][:-2] + '_%s' % string + tpp_layer[4][-2:]
                            
                            tpp_layer[1] = tpp_layer[1][:-2] + ('_%s' % string) + tpp_layer[1][-2:]
                            for tpp_line in tpp_layer:
                                proto = proto + tpp_line
                                
                        if type == 'InnerProduct':
                            fc_layer = lines[k:k+15]
                            
                            # bottom
                            fc_layer[3] = fc_layer[3][:-2] + '_%s' % string + fc_layer[3][-2:]
                            
                            # top
                            fc_layer[4] = fc_layer[4][:-2] + '_%s' % string + fc_layer[4][-2:]
                            
                            param_name = fc_layer[1][9:-2]
                            
                            fc_layer.insert(5, '  param {\n')
                            fc_layer.insert(6, '    name: "%s"\n' % (param_name + '_w'))
                            fc_layer.insert(7, '    lr_mult: 1\n')
                            fc_layer.insert(8, '  }\n')
                            
                            fc_layer.insert(9, '  param {\n')
                            fc_layer.insert(10, '    name: "%s"\n' % (param_name + '_b'))
                            fc_layer.insert(11, '    lr_mult: 1\n')
                            fc_layer.insert(12, '  }\n')
                            
                            fc_layer[1] = fc_layer[1][:-2] + ('_%s' % string) + fc_layer[1][-2:]
                            for fc_line in fc_layer:
                                proto = proto + fc_line
                                
                        if type == 'Dropout':
                            d_layer = lines[k:k+12]
                            
                            # bottom
                            d_layer[3] = d_layer[3][:-2] + '_%s' % string + d_layer[3][-2:]
                            
                            # top
                            d_layer[4] = d_layer[4][:-2] + '_%s' % string + d_layer[4][-2:]
                            
                            d_layer[1] = d_layer[1][:-2] + ('_%s' % string) + d_layer[1][-2:]
                            for d_line in d_layer:
                                proto = proto + d_line
                        
                        if type == 'Sigmoid':
                            sg_layer = lines[k:k+9]
                            
                            # bottom
                            sg_layer[3] = sg_layer[3][:-2] + '_%s' % string + sg_layer[3][-2:]
                            
                            # top
                            sg_layer[4] = sg_layer[4][:-2] + '_%s' % string + sg_layer[4][-2:]
                            
                            sg_layer.pop(7)
                            sg_layer.pop(6)
                            sg_layer.pop(5)
                            
                            sg_layer[1] = sg_layer[1][:-2] + ('_%s' % string) + sg_layer[1][-2:]
                            
                            for sg_line in sg_layer:
                                proto = proto + sg_line
                            
                            

            
        proto = proto + 'layer {\n  type: "Python"\n  name: "SoftPN"\n  bottom: "sigmoid_p1"\n  bottom: "sigmoid_p2"\n  bottom: "sigmoid_n"\n  top: "loss"\n  python_param {\n    module: "SoftPN"\n    layer: "SoftPNLossLayer"\n  }\n  loss_weight: 1\n}'
        
        
        
        return proto
    '''

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

