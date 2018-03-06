import sys
import argparse
import logging
import numpy as np
from phocnet.training.phocnet_trainer import PHOCNetTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required training parameters
    parser.add_argument('--doc_img_dir', action='store', type=str, required=True,
                      help='The location of the document images.')
    parser.add_argument('--train_annotation_file', action='store', type=str,
                      help='The file path to the READ-style XML file for the training partition of the dataset to be used.')
    parser.add_argument('--test_annotation_file', action='store', type=str, 
                      help='The file path to the READ-style XML file for the testing partition of the dataset to be used.')
    parser.add_argument('--dataset_name', action='store', type=str,
                      help='The Name of the Dataset (icfhr2016).')
    parser.add_argument('--proto_dir', action='store', type=str, required=True,
                      help='Directory where to save the protobuffer files generated during the training.')
    parser.add_argument('--lmdb_dir', action='store', type=str, required=True,
                      help='Directory where to save the LMDB databases created during training.')
    # IO parameters
    parser.add_argument('--save_net_dir', '-snd', action='store', type=str,
                      help='Directory where to save the final PHOCNet. If unspecified, the net is not saved after training')
    parser.add_argument('--recreate_lmdbs', '-rl', action='store_true', default=False,
                      help='Flag indicating to delete existing LMDBs for this dataset and recompute them.')
    parser.add_argument('--min_image_width', '-min_w', action='store', type=int, default = 26,
                      help='Minimal image width in lmdb.')
    parser.add_argument('--min_image_height', '-min_h', action='store', type=int, default = 26,
                      help='Minimal image height in lmdb.')
    parser.add_argument('--max_pixel', '-max_p', action='store', type=int, default = 300000,
                      help='maximum image height in lmdb.')


    parser.add_argument('--debug_mode', '-dm', action='store_true', default=False,
                      help='Flag indicating to run the PHOCNet training in debug mode.')
    # Caffe parameters
    parser.add_argument('--image_size', '-img_size', action='store', nargs='+', type=int,
                      help='The size of the images')
    parser.add_argument('--learning_rate', '-lr', action='store', type=float, default=0.0001, 
                      help='The learning rate for SGD training. Default: 0.0001')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                      help='The momentum for SGD training. Default: 0.9')
    parser.add_argument('--step_size', '-ss', action='store', type=int, default=70000, 
                      help='The step size at which to reduce the learning rate. Default: 70000')
    parser.add_argument('--display', action='store', type=int, default=500, 
                      help='The number of iterations after which to display the loss values. Default: 500')
    parser.add_argument('--test_interval', action='store', type=int, default=500, 
                      help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--max_iter', action='store', type=int, default=80000, 
                      help='The maximum number of SGD iterations. Default: 80000')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=10, 
                      help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                      help='The weight decay for SGD training. Default: 0.00005')
    parser.add_argument('--gamma', '-gam', action='store', type=float, default=0.1,
                       help='The value with which the learning rate is multiplied after step_size iteraionts. Default: 0.1')
    parser.add_argument('--gpu_id', '-gpu', action='store', type=int, 
                      help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    # PHOCNet parameters
    parser.add_argument('--phoc_unigram_levels', '-pul', action='store', type=lambda x: [int(elem) for elem in x.split(',')], default='2,3,4,5',
                      help='Comma seperated list of PHOC unigram levels to be used. Default: 2,3,4,5')
    parser.add_argument('--use_bigrams', '-ub', action='store_true',
                        help='Flag indicating to build the PHOC with bigrams')
    parser.add_argument('--n_train_images', '-nti', action='store', type=int, default=500000, 
                      help='The number of images to be generated for the training LMDB. Default: 500000')
    parser.add_argument('--metric', '-met', action='store', type=str, default='braycurtis',
                      help='The metric with which to compare the PHOCNet predicitions (possible metrics are all scipy metrics). Default: braycurtis')
    parser.add_argument('--annotation_delimiter', '-ad', action='store', type=str,
                      help='If the annotation in the XML is separated by special delimiters, it can be specified here.')
    parser.add_argument('--use_lower_case_only', '-ulco', action='store_true', default=False,
                      help='Flag indicating to convert all annotations from the XML to lower case before proceeding')
    
    # Dense PHOCNet parameters
    parser.add_argument('--use_dense', '-ud', action='store_true',
                        help='Flag indicating to build the PHOC with dense structure')
    parser.add_argument('--dense_net_file', action='store', type=str,
                      help='The location of the dense net file.')
    parser.add_argument('--weights', action='store', type=str,
                      help='The location of the dense net weights.')
    parser.add_argument('--pooling', '-p', action='store', type=str, default='tpp',
                      help='The type of pooling layer used before MLP')
    
    '''
    parser.add_argument('--use_tpp', '-utpp', action='store_true',
                        help='Flag indicating whether to use TPP layer')
    parser.add_argument('--use_global_pooling', '-ugp', action='store_true',
                        help='Flag indicating whether to use Global Pooling layer')
    parser.add_argument('--use_ave', '-uave', action='store_true',
                        help='Flag indicating whether to use TPP layer')
    '''
    
    
    parser.add_argument('--nblocks', '-b', action='store', type=int,
                      help='The number of dense blocks.')
    parser.add_argument('--growth_rate', '-k', action='store', type=int, 
                      help='The grwoth rate of the dense network.')
    parser.add_argument('--nlayers', '-L', action='store', type=int, 
                      help='The number of layers of the dense network.')
    parser.add_argument('--use_bottleneck', '-ubn', action='store_true',
                        help='Flag indicating whether to use bottleneck layer in dense network.')
    parser.add_argument('--no_batch_normalization', '-no_bn', action='store_true',
                        help='Flag indicating whether to use batch normalization layer in dense network.')
    parser.add_argument('--use_compression', '-uc', action='store_true',
                        help='Flag indicating whether to use compression layer in dense network.')
    parser.add_argument('--pool_init', '-pi', action='store_true',
                        help='Flag indicating whether to pooling layer before entering the first dense block.')
    parser.add_argument('--init_7', '-i7', action='store_true', default = False,
                        help='Flag indicating whether to use initial kernelsize of 7')
    parser.add_argument('--conv_init', '-ci', action='store', nargs='+', type=int, default=32,
                      help='The convolutional layers before the first block/pooling with number of feature maps.')
    parser.add_argument('--config', '-c', action='store', nargs='+', type=int,
                      help='The number convolutional layers for each block. Bottleneck layers not included')
    parser.add_argument('--dropout_ratio', '-dr', action='store', type=float, default=0.2,
                      help='The dropout ratio for the dense network. Default: 0.02')
    
    # Pruning parameters
    parser.add_argument('--max_out', '-mo', action='store', type=int, default = -1, 
                      help='Use Maxout layer. If maxout > 0: k-value.')
    parser.add_argument('--use_perm', '-up', action='store_true',
                        help='Flag indicating whether to use permutation layer network.')
    parser.add_argument('--prune', '-pr', action='store_true',
                        help='Flag indicating whether to prune inactive neurons and start retraining.')
    
    # Triplet Parameters
    parser.add_argument('--triplet', '-tri', action='store_true',
                        help='Flag indicating whether to use triplet network.')
    parser.add_argument('--triplet_string_file', '-tsf', action='store', type=str,
                      help='The location of the dense net file.')
    
        

    
    params = vars(parser.parse_args())
    
    if params["config"] is not None:
        conf = params["config"]
        
        if params["nblocks"] is None:
            params["nblocks"] = len(conf)
        elif len(conf) != params["nblocks"]:
            raise ValueError('Number of blocks does not fit configuration!')
        
        s_bn = 2 if params["use_bottleneck"] else 1
        nlayers = sum(conf)*s_bn + (params["nblocks"]-1) + 4
        
        if params["nlayers"] is None:
            params["nlayers"] = nlayers
        elif nlayers != params["nlayers"]:
            raise ValueError('Number of layers does not fit configuration! Expected: L = %i' % nlayers)
       
        
    python_path = sys.path
    if params["triplet"] and '/vol/local/install/caffe-sl/python' not in python_path:
        raise ValueError('Caffe fork for similarity learning missing. Change pythonpath!')
    
    elif not params["triplet"] and '/vol/local/install/caffe_gpu-patrec-dev/python' not in python_path:
        raise ValueError('Standard Caffe fork missing. Change pythonpath!')
        
            
        

    PHOCNetTrainer(**params).train_phocnet()

