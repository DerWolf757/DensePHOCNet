'''
Created on Jan 15, 2013

@author: lrothack
'''

from collections import defaultdict
import unicodedata
        
class ModelDefinitions(object):
    '''
    '''
    
    def __init__(self, gt_reader):
        '''
        '''
        self.__gt_reader = gt_reader
        
        
    @staticmethod
    def get_model_id(document_id, model_name, model_index):
        model_name = ''.join(c for c in model_name if c.isalnum())
        if type(model_name) == unicode:
            model_name = unicodedata.normalize('NFKD', model_name).encode('ascii',
                                                                          'ignore')
        return document_id + '_' + model_name + ('_%04d' % model_index)
    
    def get_model_spec(self, model_document, model_name, model_index):
        doc_model_dict = self.generate_document_model_definitions([model_document])
        model_id = self.get_model_id(model_document, model_name, model_index)
        model_def = doc_model_dict[model_id]
        return model_id, model_def
    
    def generate_document_model_definitions(self, document_list):
            
        doc_model_dict = {}
        for document in document_list:
            gt_list = self.__gt_reader.read_ground_truth(document_name=document)
            for counter, gt_tuple in zip(xrange(len(gt_list)), gt_list):
                m_name = gt_tuple[0]
                m_bounds = gt_tuple[1]
                m_id = self.get_model_id(document, m_name, counter)
                doc_model_dict[m_id] = (document, m_name, m_bounds)
                
        return doc_model_dict
    
    @staticmethod
    def filter_model_dict(model_dict, model_id_stoplist):
        '''
        @return: A new model_dict that does not contain any model listed in model_id_stoplist
        
        '''
        model_dict_filtered = dict((m_id, m_def) for m_id, m_def in model_dict.iteritems() 
                                   if m_id not in model_id_stoplist)
        
        return model_dict_filtered
    
    
class PatchDefinitions(object):
    '''
    '''
    
    def __init__(self, document_name, gt_reader, patch_quantizer, single_query_decoding):
        '''
        '''
        self.__document_name = document_name
        self.__gt_reader = gt_reader
        self.__patch_quantizer = patch_quantizer
        self.__single_query_decoding = single_query_decoding
        
    def generate_document_patch_definitions(self, doc_model_dict):
        '''
        Create patch generation requests consisting of a key and value:
        doc_patch_key <--> (document_id, patch_size)
        '''
        
        # Name of the document the patches are generated from
        document_id = self.__document_name 
        
        # Stores doc_patch_key for each document and patch_size
        doc_patch_dict = {}
        
        # Process queries
        for m_id, m_def in doc_model_dict.iteritems():
            # Determine quantized /snapped model bounding box
            m_bounds = m_def[2]
            # Snap bounds to dense visual word grid
            m_size_snp = self.__patch_quantizer.get_patch_size_snp(m_bounds)
            # Create individual doc_patch_keys per model for single query decoding
            # queries with the same patch size get a common id otherwise
            if self.__single_query_decoding:
                patches_document_id = document_id + '_' + m_id 
            else:
                patches_document_id = document_id

            doc_patch_key = self.get_svq_list_id(document_id=patches_document_id, patch_size=m_size_snp)
            # Patches only need to be generated if no query of the same size was processed before
            if doc_patch_key not in doc_patch_dict:
                # Register doc_patch_key and specification in order to 
                # compute the patches only once
                # The model is registered for later reference
                doc_patch_dict[doc_patch_key] = (document_id, m_size_snp, [m_id])
            else:
                # Register model in existing doc_patch_dict entry
                doc_patch_value = doc_patch_dict[doc_patch_key]
                doc_patch_value[2].append(m_id)
                        
        return doc_patch_dict 

    @staticmethod
    def get_svq_list_id(document_id, patch_size):
        return document_id + '_' + ('%03dx%03d' % patch_size)
    

class QueryTargetNumberFilter(object):
    
    def __init__(self, target_number_threshold=2):
        self.__target_num_thresh = target_number_threshold
    
    def get_model_id_stoplist(self, model_dict):
        
        #
        # Extract list of available model_names
        #
        model_name_list = [m_def[1] for m_def in model_dict.itervalues()]
        
        #
        # Build model_name histogram
        #
        model_name_hist = defaultdict(int)
        for model_name in model_name_list:
            model_name_hist[model_name] += 1
        
        #
        # Identify model_names occurring less than target_num_thresh
        #
        model_name_stoplist = (m_name for m_name in model_name_hist.iterkeys() 
                                if model_name_hist[m_name] < self.__target_num_thresh)        
        
        query_name_filter = QueryNameFilter(model_name_stoplist)
        model_id_stoplist = query_name_filter.get_model_id_stoplist(model_dict)
        
        return model_id_stoplist
    
        
class QueryNameFilter(object):
    
    def __init__(self, stop_model_names):
        self.__model_name_stoplist = stop_model_names
    
    def get_model_id_stoplist(self, model_dict):
        
        #
        # Identify model_ids representing occurrences of respective model_names
        #
        model_id_stoplist = []
        for model_name_stop in self.__model_name_stoplist:
            model_id_stoplist += [m_id for m_id in model_dict.iterkeys() 
                                  if model_dict[m_id][1] == model_name_stop]
        
        return model_id_stoplist
        
