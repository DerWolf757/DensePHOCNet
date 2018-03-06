'''
Created on Nov 28, 2017

@author: fwolf
'''

from phocnet.training.phocnet_trainer import PHOCNetTrainer

def main():
    
    
    # required training parameters
    doc_img_dir = '/vol/corpora/document-image-analysis/iam-db/images/'
    
    train_annotation_file = '/home/fwolf/Workspace/DensePHOCNet/experiments/iam/train.xml'
    test_annotation_file = '/home/fwolf/Workspace/DensePHOCNet/experiments/iam/test.xml'
    
    proto_dir = '/home/fwolf/Workspace/DensePHOCNet/data/triplet'
    
    lmdb_dir = '/data/fwolf/ordered_iam_15000'
    
    # IO parameters
    save_net_dir = '/home/fwolf/Workspace/DensePHOCNet/data/triplet'
    
    recreate_lmdbs = False
    
    debug_mode = True
    
    # Caffe parameters
    learning_rate = 0.0001
    
    momentum = 0.9
   
    step_size = 30000

    display = 100

    test_interval = 1000

    max_iter = 10000

    batch_size = 10

    weight_decay = 0.00005

    gamma = 0.1

    gpu_id = 0 
    
    # PHOCNet parameters
    phoc_unigram_levels = (2,3,4,5)
    
    use_bigrams = True
    
    n_train_images = 500000

    metric = 'cosine'
    
    annotation_delimiter = None
                      
    use_lower_case_only = False
    
    max_pixel = 100000
       
    # Dense PHOCNet parameters
    use_dense = True

    dense_net_file = None
    
    weights = '/home/fwolf/Workspace/DensePHOCNet/data/triplet/dense_L20b2k12tpp_phocnet__iam-db_nti500000_pul2-3-4-5.binaryproto'
    triplet_string_file = '/home/fwolf/Workspace/DensePHOCNet/data/triplet/triplet_base.prototxt'

    
    nblocks = 2
    growth_rate = 12
    nlayers = 20
    use_bottleneck = False
    use_compression = False
    pool_init = True
    dropout_ratio = 0.2
    use_tpp = True
    
    max_out = -1
    
    use_perm = False
    prune = False
    
    use_global_pooling = False
    
    config = (5,10)
    triplet = True
    
    


    
    
        
    PHOCNetTrainer(doc_img_dir, train_annotation_file, test_annotation_file, 
                 proto_dir, n_train_images, lmdb_dir, save_net_dir, 
                 phoc_unigram_levels, recreate_lmdbs, gpu_id, learning_rate, momentum, 
                 weight_decay, batch_size, test_interval, display, max_iter, step_size, 
                 gamma, debug_mode, metric, annotation_delimiter, use_lower_case_only,
                 use_bigrams, use_dense, use_tpp, use_global_pooling, nblocks, growth_rate, nlayers, 
                 use_bottleneck, use_compression, pool_init, dropout_ratio, max_out, use_perm, prune, config,
                 triplet, triplet_string_file, weights,
                 max_pixel = max_pixel).train_phocnet()
    
    
if __name__ == '__main__':
    main()