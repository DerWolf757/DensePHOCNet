'''
Created on Dec 11, 2017

@author: fwolf
'''
from phocnet.evaluation.phocnet_evaluator import PHOCNetEvaluation

def main():
    phocnet_bin_path = '/home/fwolf/Workspace/DensePHOCNet/data/botany/models/dense_L95b2k12_Ctppphocnet__botany_nti500000_pul2-3-4-5.binaryproto'
    
    train_xml_file = None #'/home/fwolf/Workspace/DensePHOCNet/experiments/iam/train.xml'
    
    test_xml_file = None #'/home/fwolf/Workspace/DensePHOCNet/experiments/iam/test.xml'
    
    gpu_id = 1 
    
    debug_mode = True
    
    
    doc_img_dir = None #'/vol/corpora/document-image-analysis/iam-db/images/'
    
    annotation_delimiter = None
    
    deploy_proto_path = None #'/tmp/deploy_phocnet.prototxt'
    
    metric = 'cosine'
    
    phoc_unigram_levels = (2,3,4,5)
    
    no_bigrams = False
    
    pne = PHOCNetEvaluation()
    
    dense_net = '/home/fwolf/Workspace/DensePHOCNet/data/botany/deploy/L95b2k12_Cbotany.prototxt'
    
    min_image_height = 32
    min_image_width = 32
    max_pixel = 10000
    image_size=(32,32)
    
    protocol = 'botany'
    
    
    pne.eval_qbe(phocnet_bin_path, train_xml_file, test_xml_file, 
                 gpu_id, debug_mode, doc_img_dir, annotation_delimiter, 
                 deploy_proto_path, metric, phoc_unigram_levels, no_bigrams, dense_net,
                 min_image_height, min_image_width, max_pixel, image_size,protocol)
    
    '''
    pne.eval_qbs(phocnet_bin_path, train_xml_file, test_xml_file, phoc_unigram_levels, 
                 gpu_id, debug_mode, doc_img_dir, deploy_proto_path, metric, 
                 annotation_delimiter, no_bigrams, dense_net,
                 min_image_height, min_image_width, max_pixel)
    '''
    
    
if __name__ == '__main__':
    main()