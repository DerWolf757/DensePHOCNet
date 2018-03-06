'''
Created on Aug 29, 2016

@author: ssudholt
'''
import logging
import os
import time

import cv2
import operator

import inspect

import caffe
import numpy as np
from skimage.transform import resize

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

from layers import triplet

import sys


class PHOCNetTrainer(object):
    '''
    Driver class for all PHOCNet experiments
    '''

    def __init__(self, doc_img_dir, train_annotation_file, test_annotation_file, dataset_name,
                 proto_dir, n_train_images, lmdb_dir, save_net_dir, 
                 phoc_unigram_levels, recreate_lmdbs, gpu_id, image_size, learning_rate, momentum, 
                 weight_decay, batch_size, test_interval, display, max_iter, step_size, 
                 gamma, debug_mode, metric, annotation_delimiter, use_lower_case_only,
                 use_bigrams, use_dense, pooling, nblocks, growth_rate, nlayers,
                 no_batch_normalization,
                 use_bottleneck, use_compression, pool_init, init_7, conv_init, dropout_ratio, max_out, use_perm, prune, config,
                 triplet, triplet_string_file = None,
                 dense_net_file =  None, weights = None, min_image_width = 26, min_image_height = 26, max_pixel = 300000):
        '''
        The constructor
        
        Args:
            doc_img_dir (str): the location of the document images for the given dataset
            train_annotation_file (str): the absolute path to the READ-style annotation file for the training samples
            test_annotation_file (str): the absolute path to the READ-style annotation file for the test samples
            proto_dir (str): absolute path where to save the Caffe protobuffer files
            n_train_images (int): the total number of training images to be used
            lmdb_dir (str): directory to save the LMDB files into
            save_net_dir (str): directory where to save the trained PHOCNet
            phoc_unigrams_levels (list of int): the list of unigram levels
            recreate_lmdbs (bool): whether to delete and recompute existing LMDBs
            debug_mode (bool): flag indicating to run this experiment in debug mode
            metric (str): metric for comparing the PHOCNet output during test
            annotation_delimiter (str): delimiter for the annotation in the XML files
            use_lower_case_only (bool): convert annotation to lower case before creating LMDBs
            use_bigrams (bool): if true, the PHOC predicted from the net contains bigrams
            
            gpu_id (int): the ID of the GPU to use
            learning_rate (float): the learning rate to be used in training
            momentum (float): the SGD momentum to be used in training
            weight_decay (float): the SGD weight decay to be used in training
            batch_size (int): the number of images to be used in a mini batch
            test_interval (int): the number of steps after which to evaluate the PHOCNet during training
            display (int): the number of iterations after which to show the training net loss
            max_iter (int): the maximum number of SGD iterations
            step_size (int): the number of iterations after which to reduce the learning rate
            gamma (float): the factor to multiply the step size with after step_size iterations
            
            use_dense (bool): use dense network as convolutional part
            dense_net_file (str): the prototex specification of a DenseNet
            weights (str): the caffemodel in order to train a pretrained DenseNet
        '''
        # store the experiment parameters
        self.doc_img_dir = doc_img_dir
        self.train_annotation_file = train_annotation_file
        self.test_annotation_file = test_annotation_file
        self.proto_dir = proto_dir
        self.n_train_images = n_train_images
        self.lmdb_dir = lmdb_dir
        self.save_net_dir = save_net_dir
        self.phoc_unigram_levels = phoc_unigram_levels
        self.recreate_lmdbs = recreate_lmdbs
        self.debug_mode = debug_mode
        self.metric = metric
        self.annotation_delimiter = annotation_delimiter
        self.use_lower_case_only = use_lower_case_only
        self.use_bigrams = use_bigrams
        
        # dense network specifications
        self.use_dense = use_dense
        self.dense_net_file = dense_net_file
        self.weights = weights
        self.pooling = pooling
        
        self.nblocks = nblocks
        self.growth_rate = growth_rate
        self.nlayers = nlayers
        self.use_bottleneck = use_bottleneck
        self.use_compression = use_compression
        self.pool_init = pool_init
        self.conv_init= conv_init
        self.init_7 = init_7
        self.dropout_ratio = dropout_ratio
        self.config = config
        
        self.no_batch_normalization = no_batch_normalization
        
        # triplet network
        self.triplet = triplet
        self.triplet_string_file = triplet_string_file
        
        # pruning specification
        self.max_out = max_out
        self.use_perm = use_perm
        self.prune = prune
        
        # store the Caffe parameters
        self.gpu_id = gpu_id
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.test_interval = test_interval
        self.display = display
        self.max_iter = max_iter
        self.step_size = step_size
        self.gamma = gamma        
        
        # misc members for training/evaluation
        if self.gpu_id is not None:
            self.solver_mode = 'GPU'
        else:
            self.solver_mode = 'CPU'
        self.image_size = image_size
        self.min_image_height = min_image_height
        self.min_image_width = min_image_width
        self.max_pixel = max_pixel
        self.epoch_map = None
        self.test_iter = None
        self.dataset_name = dataset_name
        
        # set up the logging
        logging_format = '[%(asctime)-19s, %(name)s] %(message)s'
        if self.debug_mode:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO
        logging.basicConfig(level=logging_level, format=logging_format)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        
        self.logger.info('--- Training Parameter: ---')
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        
        for i in args:
            self.logger.info( "%s = %s" % (i, values[i]))
        

        
        
    def train_phocnet(self):
        self.logger.info('--- Running PHOCNet Prune Training ---')
        # --- Step 1: check if we need to create the LMDBs
        # load the word lists
        #print self.dataset_name
        if self.train_annotation_file is not None and self.test_annotation_file is not None: 
            xml_reader = XMLReader(make_lower_case=self.use_lower_case_only)
            self.dataset_name, train_list, test_list = xml_reader.load_train_test_xml(train_xml_path=self.train_annotation_file, 
                                                                                      test_xml_path=self.test_annotation_file, 
                                                                                      img_dir=self.doc_img_dir)
        elif self.dataset_name is not None:
            train_list, test_list, qry_list = DatasetLoader.load_icfhr2016_competition('botany',
                                                                                       train_set='Train_III',
                                                                                       path='/vol/corpora/document-image-analysis/competition_icfhr2016/')
            
        else:
            self.logger.info('Annotation missing')
        
        phoc_unigrams = unigrams_from_word_list(word_list=train_list, split_character=self.annotation_delimiter)
        self.logger.info('PHOC unigrams: %s', ' '.join(phoc_unigrams))
        self.test_iter = len(test_list)
        self.logger.info('Using dataset \'%s\'', self.dataset_name)
        
        lmdb_prefix = '%s_nti%d_pul%s' % (self.dataset_name, self.n_train_images,
                                              '-'.join([str(elem) for elem in self.phoc_unigram_levels]))
        train_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_word_images_lmdb' % lmdb_prefix)
        train_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_train_phocs_lmdb' % lmdb_prefix)
        test_word_images_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_word_images_lmdb' % lmdb_prefix)
        test_phoc_lmdb_path = os.path.join(self.lmdb_dir, '%s_test_phocs_lmdb' % lmdb_prefix)
            
        # check if we need to create LMDBs
        if not self.triplet:
            lmdbs_exist = (os.path.exists(train_word_images_lmdb_path),
                           os.path.exists(train_phoc_lmdb_path),
                           os.path.exists(test_word_images_lmdb_path),
                           os.path.exists(test_phoc_lmdb_path))
        else:
            lmdbs_exist = os.path.exists(train_word_images_lmdb_path)
            
                        
        if self.use_bigrams:
            n_bigrams = 50
            bigrams = get_most_common_n_grams(words=[word.get_transcription() 
                                                     for word in train_list], 
                                              num_results=n_bigrams, n=2)
            bigram_levels = [2]
        else:       
            n_bigrams = 0         
            bigrams = None
            bigram_levels = None        
        if not np.all(lmdbs_exist) or self.recreate_lmdbs:     
            self.logger.info('Creating LMDBs...')  
                    
            if not self.triplet:
                train_phocs = build_phoc(words=[word.get_transcription() for word in train_list], 
                                         phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                         phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                         split_character=self.annotation_delimiter,
                                         on_unknown_unigram='warn')
                test_phocs = build_phoc(words=[word.get_transcription() for word in test_list],
                                        phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                        phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                        split_character=self.annotation_delimiter,
                                        on_unknown_unigram='warn')
                self._create_train_test_phocs_lmdbs(train_list=train_list, train_phocs=train_phocs, 
                                                    test_list=test_list, test_phocs=test_phocs,
                                                    train_word_images_lmdb_path=train_word_images_lmdb_path,
                                                    train_phoc_lmdb_path=train_phoc_lmdb_path,
                                                    test_word_images_lmdb_path=test_word_images_lmdb_path,
                                                    test_phoc_lmdb_path=test_phoc_lmdb_path,image_size = self.image_size)
            else:
                train_phocs = build_phoc(words=[word.get_transcription() for word in train_list], 
                                         phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                         phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                         split_character=self.annotation_delimiter,
                                         on_unknown_unigram='warn')
                test_phocs = build_phoc(words=[word.get_transcription() for word in test_list],
                                        phoc_unigrams=phoc_unigrams, unigram_levels=self.phoc_unigram_levels,
                                        phoc_bigrams=bigrams, bigram_levels=bigram_levels,
                                        split_character=self.annotation_delimiter,
                                        on_unknown_unigram='warn')
                self._create_ordered_train_test_phocs_lmdbs(train_list=train_list, train_phocs=train_phocs, 
                                                            test_list=test_list, test_phocs=test_phocs,
                                                            train_word_images_lmdb_path=train_word_images_lmdb_path,
                                                            train_phoc_lmdb_path=train_phoc_lmdb_path,
                                                            test_word_images_lmdb_path=test_word_images_lmdb_path,
                                                            test_phoc_lmdb_path=test_phoc_lmdb_path)
        else:
            self.logger.info('Found LMDBs...')
        
        # --- Step 2: create the proto files
        self.logger.info('Saving proto files...')
        # prepare the output paths
        
        
        name = self.get_net_name()
        
        train_proto_path = os.path.join(self.proto_dir, 'train_%s_%s.prototxt' % (name,self.dataset_name))
        test_proto_path = os.path.join(self.proto_dir, 'test_%s_%s.prototxt' % (name,self.dataset_name))
        solver_proto_path = os.path.join(self.proto_dir, 'solver_%s_%s.prototxt' % (name,self.dataset_name))
      
        if not self.triplet:
            # generate the proto files
            n_attributes = np.sum(self.phoc_unigram_levels)*len(phoc_unigrams)
            if self.use_bigrams:
                n_attributes += np.sum(bigram_levels)*n_bigrams
            mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=self.gpu_id is not None)
            
            
            if self.use_dense:    
                train_proto = mpg.get_dense_phocnet(word_image_lmdb_path=train_word_images_lmdb_path, phoc_lmdb_path=train_phoc_lmdb_path, 
                                              phoc_size=n_attributes, generate_deploy=False,
                                              nblocks = self.nblocks, growth_rate = self.growth_rate,
                                              nlayers = self.nlayers, config = self.config,
                                              no_batch_normalization = self.no_batch_normalization, 
                                              use_bottleneck = self.use_bottleneck, use_compression = self.use_compression, 
                                              pool_init = self.pool_init, conv_init = self.conv_init, init_7 = self.init_7, dropout_ratio = 0.0, dense_net_file = self.dense_net_file,
                                              pooling = self.pooling, max_out = self.max_out, use_perm = self.use_perm)
                    
                test_proto = mpg.get_dense_phocnet(word_image_lmdb_path=test_word_images_lmdb_path, phoc_lmdb_path=test_phoc_lmdb_path, 
                                              phoc_size=n_attributes, generate_deploy=False,
                                              nblocks = self.nblocks, growth_rate = self.growth_rate,
                                              nlayers = self.nlayers, config = self.config,
                                              no_batch_normalization = self.no_batch_normalization,
                                              use_bottleneck = self.use_bottleneck, use_compression = self.use_compression, 
                                              pool_init = self.pool_init, conv_init = self.conv_init, init_7 = self.init_7, dropout_ratio = 0.0, dense_net_file = self.dense_net_file,
                                              pooling = self.pooling, max_out = self.max_out, use_perm = self.use_perm)
            else:
                train_proto = mpg.get_phocnet(word_image_lmdb_path=train_word_images_lmdb_path, phoc_lmdb_path=train_phoc_lmdb_path, 
                                              phoc_size=n_attributes, pooling = self.pooling, 
                                              generate_deploy=False, max_out = self.max_out, use_perm = self.use_perm)
                test_proto = mpg.get_phocnet(word_image_lmdb_path=test_word_images_lmdb_path, phoc_lmdb_path=test_phoc_lmdb_path, 
                                             phoc_size=n_attributes, pooling = self.pooling,
                                             generate_deploy=False, max_out = self.max_out, use_perm = self.use_perm)
                
                
            
            
        else:
            mpg = ModelProtoGenerator(initialization='msra', use_cudnn_engine=self.gpu_id is not None)
            train_proto = mpg.get_triplet_net(train_word_images_lmdb_path, self.nblocks, self.growth_rate, self.nlayers, 
                                              self.config, self.no_batch_normalization, self.use_bottleneck, self.use_compression, 
                                              self.pool_init, self.dropout_ratio)
            
            test_proto = mpg.get_triplet_net(test_word_images_lmdb_path, self.nblocks, self.growth_rate, self.nlayers, 
                                              self.config, self.no_batch_normalization, self.use_bottleneck, self.use_compression, 
                                              self.pool_init, self.dropout_ratio)
            
            
        
    
        # save the proto files
        save_prototxt(file_path=train_proto_path, proto_object=train_proto, header_comment='Train PHOCNet %s' % self.dataset_name)
        save_prototxt(file_path=test_proto_path, proto_object=test_proto, header_comment='Test PHOCNet %s' % self.dataset_name)
        
        if self.triplet:
            solver_proto = generate_solver_proto(train_net=train_proto_path, 
                                                 base_lr=self.learning_rate, momentum=self.momentum, display=self.display,
                                                 lr_policy='step', gamma=self.gamma, stepsize=self.step_size,
                                                 solver_mode=self.solver_mode, iter_size=self.batch_size, max_iter=self.max_iter,
                                                 average_loss=self.display,
                                                 test_initialization = False if self.triplet else True,
                                                 weight_decay=self.weight_decay)
        else:
            solver_proto = generate_solver_proto(train_net=train_proto_path, test_net=test_proto_path,
                                                 base_lr=self.learning_rate, momentum=self.momentum, display=self.display,
                                                 lr_policy='step', gamma=self.gamma, stepsize=self.step_size,
                                                 solver_mode=self.solver_mode, iter_size=self.batch_size, max_iter=self.max_iter,
                                                 average_loss=self.display, test_iter=self.test_iter, test_interval=self.test_interval, 
                                                 weight_decay=self.weight_decay)
            
        
        save_prototxt(file_path=solver_proto_path, proto_object=solver_proto, header_comment='Solver PHOCNet %s' % self.dataset_name)
        
        # --- Step 3: train the PHOCNet
        self.logger.info('Starting SGD for 0 pruned Neurons')
        solver = self._run_sgd(solver_proto_path=solver_proto_path)
        
        #size_tpp = solver.net.blobs['tpp5'].data.shape[1]
        
        # --- Step 4: prune the PHOCNet
        #k = self.max_out
        
        '''
        if k > 0:
            self.accuracies_all_nets = np.zeros(k)
            self.accuracies_all_nets[0] = self.epoch_map[-1]
            
            
            if self.prune:  
                self.logger.info('Accuricy for %i pruned Neurons: %f' % (0, self.accuracies_all_nets[0]))
                
                for pruned in range(k-1):
                    self.logger.info('Pruning layer')
                    solver = self.neuron_pruning(solver, size_tpp, k, pruned+1)
                    self.logger.info('Training network, pruning # %i' % (pruned + 1)) 
                    solver = self.__run_solver(solver)
                    
                    self.accuracies_all_nets[pruned + 1 ] = self.epoch_map[-1]
                    self.logger.info('Accuricy for %i pruned Neurons: %f' % (pruned+1, self.accuracies_all_nets[pruned + 1]))
                    
                    # run post train callback
                    self.logger.info('Running post-train evaluation')
                    
                    # if self.save_net is not None, save the PHOCNet to the desired location
                    if self.save_net_dir is not None:
                        name = self.get_net_name()
                        filename = name + '_%s_nti%d_pul%s(%i).binaryproto' % (self.dataset_name, self.n_train_images,
                                                                   '-'.join([str(elem) for elem in self.phoc_unigram_levels]), pruned+1)
                        solver.net.save(os.path.join(self.save_net_dir, filename))
                    # return the solver
        '''
            
            
        

    def pretrain_callback(self, solver):
        '''
        Method called before starting the training
        '''        
        # init numpy arrays for mAP results        
        epochs = self.max_iter/self.test_interval
        self.epoch_map = np.zeros(epochs+1)
        self.epoch_map[0], _ = calc_map_from_cnn_features(solver=solver, 
                                                          test_iterations=self.test_iter, 
                                                          metric=self.metric)
        self.logger.info('mAP: %f', self.epoch_map[0])
    
    def test_callback(self, solver, epoch):
        '''
        Method called every self.test_interval iterations during training
        '''
        self.logger.info('Evaluating CNN after %d steps:', epoch*solver.param.test_interval)
        self.epoch_map[epoch+1], _ = calc_map_from_cnn_features(solver=solver, 
                                                                test_iterations=self.test_iter, 
                                                                metric=self.metric)
        self.logger.info('mAP: %f', self.epoch_map[epoch+1])
    
    def posttrain_callback(self, solver):
        '''
        Method called after finishing the training
        '''
        # if self.save_net is not None, save the PHOCNet to the desired location
        if self.save_net_dir is not None:
            
            name = self.get_net_name()
            filename = name + '_%s_nti%d_pul%s.binaryproto' % (self.dataset_name, self.n_train_images,
                                                               '-'.join([str(elem) for elem in self.phoc_unigram_levels]))
            solver.net.save(os.path.join(self.save_net_dir, filename))
            
    def save_snapshot(self, solver, epoch):
        '''
        Method called every self.test_interval iterations during training
        '''
        self.logger.info('Evaluating CNN after %d steps:', (epoch+1)*solver.param.test_interval)
        
        if self.save_net_dir is not None:
            
            name = self.get_net_name()
            filename = name + '_%s_nti%d_pul%s_it%i.binaryproto' % (self.dataset_name, self.n_train_images,
                                                               '-'.join([str(elem) for elem in self.phoc_unigram_levels]), (epoch+1)*self.test_interval)
            solver.net.save(os.path.join(self.save_net_dir, filename))
            
            
    
    def _create_train_test_phocs_lmdbs(self, train_list, train_phocs, test_list, test_phocs, 
                                       train_word_images_lmdb_path, train_phoc_lmdb_path,
                                       test_word_images_lmdb_path, test_phoc_lmdb_path,image_size):
        start_time = time.time()        
        # --- TRAIN IMAGES
        # find all unique transcriptions and the label map...
        _, transcription_map = self.__get_unique_transcriptions_and_labelmap(train_list, test_list)
        # get the numeric training labels plus a random order to insert them into
        # create the numeric labels and counts
        train_labels = np.array([transcription_map[word.get_transcription()] for word in train_list])
        unique_train_labels, counts = np.unique(train_labels, return_counts=True)
        
        
        # find the number of images that should be present for training per class
        n_images_per_class = self.n_train_images/unique_train_labels.shape[0] + 1
        # create randomly shuffled numbers for later use as keys
        random_indices = list(xrange(n_images_per_class*unique_train_labels.shape[0]))
        np.random.shuffle(random_indices)
                
        
        #set random limits for affine transform
        random_limits = (0.8, 1.1)
        n_rescales = 0
        
        # loading should be done in gray scale
        load_grayscale = True
        
        # create train LMDB  
        self.logger.info('Creating Training LMDB (%d total word images)', len(random_indices))      
        lmdb_creator = CaffeLMDBCreator()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=train_word_images_lmdb_path, 
                                              additional_lmdb_path=train_phoc_lmdb_path,
                                              create=True)
        
        for cur_label, count in zip(unique_train_labels, counts):
            # find the words for the current class label and the
            # corresponding PHOC            
            cur_word_indices = np.where(train_labels == cur_label)[0]  
            cur_transcription = train_list[cur_word_indices[0]].get_transcription()
            cur_phoc = NumpyHelper.get_unique_rows(train_phocs[cur_word_indices])
            # unique rows should only return one specific PHOC
            if cur_phoc.shape[0] != 1:
                raise ValueError('Extracted more than one PHOC for label %d' % cur_label)
            cur_phoc = np.atleast_3d(cur_phoc).transpose((2,0,1)).astype(np.uint8)
                      
            # if there are to many images for the current word image class, 
            # draw from them and cut the rest off
            if count > n_images_per_class:
                np.random.shuffle(cur_word_indices)
                cur_word_indices = cur_word_indices[:n_images_per_class]
            # load the word images
            cur_word_images = []            
            for idx in cur_word_indices:                
                img = train_list[idx].get_word_image(gray_scale=load_grayscale)  
                # check image size
                img, resized = self.__check_size(img,image_size)
                
                n_rescales += int(resized)
                
                # append to the current word images and
                # put into LMDB
                cur_word_images.append(img)
                key = '%s_%s' % (str(random_indices.pop()).zfill(8), cur_transcription.encode('ascii', 'ignore'))                
                lmdb_creator.put_dual(img_mat=np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8), 
                                      additional_mat=cur_phoc, label=cur_label, key=key)
                            
            # extract the extra augmented images
            # the random limits are the maximum percentage
            # that the destination point may deviate from the reference point
            # in the affine transform            
            if len(cur_word_images) < n_images_per_class:
                # create the warped images
                inds = np.random.randint(len(cur_word_images), size=n_images_per_class - len(cur_word_images))                
                for ind in inds:
                    aug_img = AugmentationCreator.create_affine_transform_augmentation(img=cur_word_images[ind], random_limits=random_limits)
                    aug_img = np.atleast_3d(aug_img).transpose((2,0,1)).astype(np.uint8)
                    key = '%s_%s' % (str(random_indices.pop()).zfill(8), cur_transcription.encode('ascii', 'ignore'))
                    lmdb_creator.put_dual(img_mat=aug_img, additional_mat=cur_phoc, label=cur_label, key=key)
        # wrap up training LMDB creation
        if len(random_indices) != 0:
            raise ValueError('Random Indices are not empty, something went wrong during training LMDB creation')
        lmdb_creator.finish_creation()
        # write the label map to the LMDBs as well
        '''
        write_list(file_path=train_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=train_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        '''
        self.logger.info('Finished processing train words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
        
        
        # --- TEST IMAGES
        self.logger.info('Creating Test LMDB (%d total word images)', len(test_list))
        n_rescales = 0
        start_time = time.time()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=test_word_images_lmdb_path, additional_lmdb_path=test_phoc_lmdb_path, 
                                              create=True, label_map=transcription_map)
        for word, phoc in zip(test_list, test_phocs): 
            if word.get_transcription() not in transcription_map:
                transcription_map[word.get_transcription()] = len(transcription_map)
            img = word.get_word_image(gray_scale=load_grayscale)
            img, resized = self.__check_size(img,image_size)
            if img is None:
                    self.logger.warning('!WARNING! Found image with 0 width or height!')
            else:
                n_rescales += int(resized)
                img = np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8)
                phoc_3d = np.atleast_3d(phoc).transpose((2,0,1)).astype(np.uint8)
                lmdb_creator.put_dual(img_mat=img, additional_mat=phoc_3d, label=transcription_map[word.get_transcription()])
        lmdb_creator.finish_creation()
        '''
        write_list(file_path=test_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=test_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        '''
        
        self.logger.info('Finished processing test words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
        
    def _create_ordered_train_test_phocs_lmdbs(self, train_list, train_phocs, test_list, test_phocs, 
                                               train_word_images_lmdb_path, train_phoc_lmdb_path,
                                               test_word_images_lmdb_path, test_phoc_lmdb_path):
        start_time = time.time()        
        # --- TRAIN IMAGES
        # find all unique transcriptions and the label map...
        _, transcription_map = self.__get_unique_transcriptions_and_labelmap(train_list, test_list)
        # get the numeric training labels plus a random order to insert them into
        # create the numeric labels and counts
        train_labels = np.array([transcription_map[word.get_transcription()] for word in train_list])
        unique_train_labels, counts = np.unique(train_labels, return_counts=True)
        
        
        # find the number of images that should be present for training per class
        n_images_per_class = self.n_train_images/unique_train_labels.shape[0] + 1
        # create randomly shuffled numbers for later use as keys
        random_indices = list(xrange(n_images_per_class*unique_train_labels.shape[0],0,-1))
        #np.random.shuffle(random_indices)
                
        
        #set random limits for affine transform
        random_limits = (0.8, 1.1)
        n_rescales = 0
        
        # loading should be done in gray scale
        load_grayscale = True
        
        # create train LMDB  
        self.logger.info('Creating Training LMDB (%d total word images)', len(random_indices))      
        lmdb_creator = CaffeLMDBCreator()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=train_word_images_lmdb_path, 
                                              additional_lmdb_path=train_phoc_lmdb_path,
                                              create=True)
        
        for cur_label, count in zip(unique_train_labels, counts):
            # find the words for the current class label and the
            # corresponding PHOC            
            cur_word_indices = np.where(train_labels == cur_label)[0]  
            cur_transcription = train_list[cur_word_indices[0]].get_transcription()
            cur_phoc = NumpyHelper.get_unique_rows(train_phocs[cur_word_indices])
            # unique rows should only return one specific PHOC
            if cur_phoc.shape[0] != 1:
                raise ValueError('Extracted more than one PHOC for label %d' % cur_label)
            cur_phoc = np.atleast_3d(cur_phoc).transpose((2,0,1)).astype(np.uint8)
                      
            # if there are to many images for the current word image class, 
            # draw from them and cut the rest off
            if count > n_images_per_class:
                np.random.shuffle(cur_word_indices)
                cur_word_indices = cur_word_indices[:n_images_per_class]
            # load the word images
            cur_word_images = []            
            for idx in cur_word_indices:                
                img = train_list[idx].get_word_image(gray_scale=load_grayscale)  
                # check image size
                img, resized = self.__check_size(img)
                
                n_rescales += int(resized)
                
                # append to the current word images and
                # put into LMDB
                cur_word_images.append(img)
                key = '%s_%i' % (str(random_indices.pop()).zfill(8), n_images_per_class)

                 
                lmdb_creator.put_dual(img_mat=np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8), 
                                      additional_mat=cur_phoc, label=cur_label, key=key)
                            
            # extract the extra augmented images
            # the random limits are the maximum percentage
            # that the destination point may deviate from the reference point
            # in the affine transform            
            if len(cur_word_images) < n_images_per_class:
                # create the warped images
                inds = np.random.randint(len(cur_word_images), size=n_images_per_class - len(cur_word_images))                
                for ind in inds:
                    aug_img = AugmentationCreator.create_affine_transform_augmentation(img=cur_word_images[ind], random_limits=random_limits)
                    aug_img = np.atleast_3d(aug_img).transpose((2,0,1)).astype(np.uint8)
                    key = '%s_%i' % (str(random_indices.pop()).zfill(8), n_images_per_class)
                    
                    lmdb_creator.put_dual(img_mat=aug_img, additional_mat=cur_phoc, label=cur_label, key=key)
        # wrap up training LMDB creation
        if len(random_indices) != 0:
            raise ValueError('Random Indices are not empty, something went wrong during training LMDB creation')
        lmdb_creator.finish_creation()
        # write the label map to the LMDBs as well        
        write_list(file_path=train_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=train_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        self.logger.info('Finished processing train words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
        
        
        # --- TEST IMAGES
        self.logger.info('Creating Test LMDB (%d total word images)', len(test_list))
        n_rescales = 0
        start_time = time.time()
        lmdb_creator.open_dual_lmdb_for_write(image_lmdb_path=test_word_images_lmdb_path, additional_lmdb_path=test_phoc_lmdb_path, 
                                              create=True, label_map=transcription_map)
        for word, phoc in zip(test_list, test_phocs): 
            if word.get_transcription() not in transcription_map:
                transcription_map[word.get_transcription()] = len(transcription_map)
            img = word.get_word_image(gray_scale=load_grayscale)
            img, resized = self.__check_size(img)
            if img is None:
                    self.logger.warning('!WARNING! Found image with 0 width or height!')
            else:
                n_rescales += int(resized)
                img = np.atleast_3d(img).transpose((2,0,1)).astype(np.uint8)
                phoc_3d = np.atleast_3d(phoc).transpose((2,0,1)).astype(np.uint8)
                lmdb_creator.put_dual(img_mat=img, additional_mat=phoc_3d, label=transcription_map[word.get_transcription()])
        lmdb_creator.finish_creation()
        write_list(file_path=test_word_images_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        write_list(file_path=test_phoc_lmdb_path + '/label_map.txt', 
                   line_list=['%s %s' % elem for elem in transcription_map.items()])
        self.logger.info('Finished processing test words (took %s, %d rescales)', convert_secs2HHMMSS(time.time() - start_time), n_rescales)
    
    def __check_size(self, img, image_size):
        '''
        checks if the image accords to the minimum and maximum size requirements
        
        Returns:
            tuple (img, bool):
                 img: the original image if the image size was ok, a resized image otherwise
                 bool: flag indicating whether the image was resized
        '''
        
        if image_size is None:
            # check minimal size
            scale_height = float(self.min_image_height+1)/float(img.shape[0])
            scale_width = float(self.min_image_width+1)/float(img.shape[1])
            
            if img.shape[0] < self.min_image_height and scale_height*img.shape[1] > self.min_image_width:
                resized = True
                new_shape = (int(scale_height*img.shape[0]), int(scale_height*img.shape[1]))
            elif img.shape[1] < self.min_image_width and scale_width*img.shape[0] > self.min_image_height:
                resized = True
                new_shape = (int(scale_width*img.shape[0]), int(scale_width*img.shape[1]))
            else:
                resized = False
                
                
            # check maximum image size
            if img.shape[0]*img.shape[1] > self.max_pixel:
                resized = True
                relation = float(img.shape[0])/float(img.shape[1])
                
                i0 = np.sqrt(self.max_pixel/relation)
                i1 = i0*relation
                
                new_shape = (int(i1), int(i0))
        else:
            resized = True
            new_shape = (image_size[0], image_size[1])

        
        if resized:
            new_img = resize(image=img, output_shape=new_shape)
            new_img = (new_img*255).astype('uint8')
            
            return new_img, resized
        else:
            return img, resized
           
    
    def __get_unique_transcriptions_and_labelmap(self, train_list, test_list):
        '''
        Returns a list of unique transcriptions for the given train and test lists
        and creates a dictionary mapping transcriptions to numeric class labels.
        '''
        unique_transcriptions = [word.get_transcription() for word in train_list]
        unique_transcriptions.extend([word.get_transcription() for word in test_list])
        unique_transcriptions = list(set(unique_transcriptions))
        
        transcription_map = dict((k,v) for v,k in enumerate(unique_transcriptions))
        return unique_transcriptions, transcription_map        
    
    def _run_sgd(self, solver_proto_path):
        '''
        Starts the SGD training of the PHOCNet
        
        Args:
            solver_proto_path (str): the absolute path to the solver protobuffer file to use
        '''
        # Set CPU/GPU mode for solver training
        if self.gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', self.gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()
        
        # Create SGD solver
        self.logger.info('Using solver protofile at %s', solver_proto_path)
        solver = self.__get_solver(solver_proto_path)        
        epochs = self.max_iter/self.test_interval
        
        # run test on the net before training
        self.logger.info('Running pre-train evaluation')
        
        if not self.triplet:
            self.pretrain_callback(solver=solver)
        
        # run the training
        self.logger.info('Finished Setup, running SGD')        
        for epoch in xrange(epochs):
            # run training until we want to test
            self.__solver_step(solver, self.test_interval)
            
            # run test callback after test_interval iterations
            self.logger.debug('Running test evaluation')
            if not self.triplet:
                self.test_callback(solver=solver, epoch=epoch)
            elif self.triplet:
                self.save_snapshot(solver=solver, epoch=epoch)
                
        # if we have iterations left to compute, do so
        iters_left = self.max_iter % self.test_interval
        if iters_left > 0:
            self.__solver_step(solver, iters_left)
            
        # run post train callback
        self.logger.info('Running post-train evaluation')
        
        if not self.triplet:
            self.posttrain_callback(solver=solver)
            
        # return the solver
        return solver
    
    def __solver_step(self, solver, steps):
        '''
        Runs Caffe solver suppressing Caffe output if necessary
        '''
        if not self.debug_mode:
            with Suppressor():
                solver.step(steps)
        else:
            solver.step(steps)
    
    def __get_solver(self, solver_proto_path):
        '''
        Returns a caffe.SGDSolver for the given protofile path,
        ignoring Caffe command line chatter if debug mode is not set
        to True.
        '''
        if not self.debug_mode:
            # disable Caffe init chatter when not in debug
            with Suppressor():
                solver = caffe.SGDSolver(solver_proto_path)
                
        else:
            solver =  caffe.SGDSolver(solver_proto_path)

        
        # use DenseNet weights if available          
        if self.weights is not None:
            if not self.triplet:
                solver.net.copy_from(self.weights)
            else:
                self.logger.info('Initializing Weights')
                base_net = caffe.Net(self.triplet_string_file, caffe.TEST)
                base_net.copy_from(self.weights)
                
                
                print 'Initializing fc network...'
                param_fc6 = base_net.params["fc6_d"]
                for k in range(len(param_fc6)):
                    solver.net.params['fc6_d_p1'][k].data[...] = param_fc6[k].data
                    solver.net.params['fc6_d_p2'][k].data[...] = param_fc6[k].data
                    solver.net.params['fc6_d_n'][k].data[...] = param_fc6[k].data
                    
                param_fc7 = base_net.params["fc7_d"]
                for k in range(len(param_fc7)):
                    solver.net.params['fc7_d_p1'][k].data[...] = param_fc7[k].data
                    solver.net.params['fc7_d_p2'][k].data[...] = param_fc7[k].data
                    solver.net.params['fc7_d_n'][k].data[...] = param_fc7[k].data
                    
                param_fc8 = base_net.params["fc8_d"]
                for k in range(len(param_fc8)):
                    solver.net.params['fc8_d_p1'][k].data[...] = param_fc8[k].data
                    solver.net.params['fc8_d_p2'][k].data[...] = param_fc8[k].data
                    solver.net.params['fc8_d_n'][k].data[...] = param_fc8[k].data
                    
        return solver
    
    def __run_solver(self,solver):
        # Set CPU/GPU mode for solver training
        if self.gpu_id != None:
            self.logger.info('Setting Caffe to GPU mode using device %d', self.gpu_id)
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_id)
        else:
            self.logger.info('Setting Caffe to CPU mode')
            caffe.set_mode_cpu()
               
        epochs = self.max_iter/self.test_interval
        
        # run test on the net before training
        self.logger.info('Running pre-train evaluation')
        self.pretrain_callback(solver=solver)
        
        # run the training
        self.logger.info('Finished Setup, running SGD')        
        for epoch in xrange(epochs):
            # run training until we want to test
            self.__solver_step(solver, self.test_interval)
            
            # run test callback after test_interval iterations
            self.logger.debug('Running test evaluation')
            self.test_callback(solver=solver, epoch=epoch)
        # if we have iterations left to compute, do so
        iters_left = self.max_iter % self.test_interval
        if iters_left > 0:
            self.__solver_step(solver, iters_left)
            
        
        
        return solver
    
    
    
    def neuron_pruning(self, solver_net, size_tpp, k, pruned):
        neurons_relevance = self.count_act_neurons(solver_net, size_tpp, k)
        
        relevance_sort = np.argsort(neurons_relevance, axis = 1)
        
        args_min = np.transpose(relevance_sort[:,:pruned])

 
        Scale_new = np.ones(solver_net.net.blobs['premax'].data.shape)
        Scale_new[0,range(np.shape(Scale_new)[1]),args_min,0] = 0
        
        solver_net.net.params['premax'][0].data[...] = Scale_new
        
        return solver_net

    def count_act_neurons(self, solver_net, size_tpp, k):
        # At the end run over training set  
        blob_count = np.zeros( (int(size_tpp/k), k ))
        
        self.logger.info('Count Active Neurons')        
        for item_num in range(self.n_train_images/10):
            solver_net.net.forward()
 
            premax_data = solver_net.net.blobs['premax'].data.copy()
            
            data_b = premax_data[0,:,:,0]
            argM = np.argmax(data_b, axis = 1)
            
            for a in range(len(argM)):
                if False:
                    for neuron in range(k):
                        blob_count[a,neuron] +=1
                        continue
                else:
                    if argM[a]==0:
                        blob_count[a,0] +=1
                        continue
                    if argM[a]==1:
                        blob_count[a,1] +=1
                        continue
                    if argM[a]==2:
                        blob_count[a,2] +=1
                        continue
                    if argM[a]==3:
                        blob_count[a,3] +=1
                        continue
                  
            if item_num % 5000 == 0:
                self.logger.info('   Finished %i ndarrays' % item_num)      
        
        return blob_count
    
    def get_net_name(self):
        #get name
        str_triplet = 'triplet' if self.triplet else ''
        str_dense = 'dense_' if self.use_dense else ''
        str_pooling = self.pooling
        str_conf = ''
        str_prune = 'mo%i' % self.max_out if self.max_out >= 0 else ''
        
        if self.prune:
            str_prune = str_prune + 'p'
            
            
        if self.dense_net_file is None and self.use_dense:
            str_B = 'B' if self.use_bottleneck else ''
            str_C = 'C' if self.use_compression else ''
            str_BC = ''
            
            if (str_B + str_C):
                str_BC = '_'+str_B+str_C
                
            str_conf = ('L%ib%ik%i' % (self.nlayers,self.nblocks,self.growth_rate))+str_BC
            
        else:
            str_conf = 'ff'
            
        name = '%s%s%s%sphocnet_%s' % (str_triplet, str_dense, str_conf,str_pooling, str_prune)
        
        return name
        
