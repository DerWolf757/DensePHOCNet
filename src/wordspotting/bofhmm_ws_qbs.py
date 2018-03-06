'''
Created on Apr 22, 2014

@author: leonard
'''
import logging
from os.path import exists
from os import makedirs
import itertools
from collections import defaultdict
import cPickle as pickle
import numpy as np
from bofhwr.wordspotting.bofhmm_ws import Wordspotting, WhitespaceModels
from bofhwr.hmm.mm_reader import MMCharModelReader, ConceptDefinitions
from bofhwr.wordspotting.query_results import QueryResultsEvaluator
from bofhwr.features.visualwords_io import VisualWordsIO
from bofhwr.wordspotting.gt_reader import GroundTruthReader
from bofhwr.wordspotting.definitions import ModelDefinitions
from bofhwr.wordspotting.model import  ModelSVQGenerator
from bofhwr.hmm.mm_estimation import CharModelEstimator, CharModelGenerator
from bofhwr.features.patch_termvector_sequence import PatchTermvectorGenerator as PatchTermGen, \
    PatchQuantizer, PatchSequenceGenerator
from bofhwr.features.om_termvector_processor import PatchTermvectorOutputModelTransformGenerator as PatchTermOMTransformGen
from esmeralda.es_base import MMBase
from esmeralda.svq import SlidingProbTermVector, SVQVectors, SVQVectorsIO
from esmeralda.align import MMHypChain, MMAlign, MMConcept, MMDefinition
from esmeralda.cmd.es_hmm import MMConceptDef, MMInit
from wis.mser.regions import RegionHypotheses

class QueryByStringTraining(object):

    def __init__(self, config, char_model_id):
        self.__config = config
        self.__char_model_id = char_model_id


        self.__gt_reader = GroundTruthReader(config.data_base,
                                             gtp_encoding=config.gtp_encoding)

    def __train_spec_list(self, document_list):

        train_spec_list = []
        for document_id in document_list:
            gt_list = self.__gt_reader.read_ground_truth(document_id)
            train_spec_list += [(document_id, gt_tup[0], gt_index) for gt_index, gt_tup in enumerate(gt_list)]

        return train_spec_list


    def training(self, document_list):
        logger = logging.getLogger('QueryByStringTraining::training')
        train_spec_list = self.__train_spec_list(document_list)
        cm_init = CharModelInitializer(self.__config,
                                       self.__char_model_id)
        document_models_dict = cm_init.to_document_models_dict(train_spec_list)
        cm_size_dict_list = cm_init.estimate_character_models(document_models_dict)
        # Print char model size estimates for debugging
        # Only consider size estimates for the last Baum-Welch iteration
        cm_size_dict = cm_size_dict_list[-1]
        char_key_sorter = CharModelGenerator.context_model_to_model
        char_key_list = sorted(cm_size_dict.keys(), key=char_key_sorter)
        cm_stat_str = '\n'.join(['{0}: ({1}, {2}) '.format(c_key,
                                                           *cm_size_dict[c_key])
                                                           for c_key in char_key_list])

        logger.info('Estimated character model dict '
                    '(model state index = %d):\n%s',
                    len(cm_size_dict_list) - 1,
                    cm_stat_str)
        return cm_size_dict_list

    def alignment_forced(self, document_list):
        train_spec_list = self.__train_spec_list(document_list)
        cm_init = CharModelInitializer(self.__config,
                                       self.__char_model_id)
        document_models_dict = cm_init.to_document_models_dict(train_spec_list)
        cm_def_list = cm_init.align_character_models(document_models_dict)
        return cm_def_list

class QueryByStringRetrieval(object):

    def __init__(self, config, char_model_id):
        self.__config = config
        self.__char_model_id = char_model_id
        char_model_name = config.model_name
        concept_id_list = [QueryByString.QUERIES_CONCEPT_ID]
        self.__mm_concept_reader = MMCharModelReader(config,
                                                     char_model_name,
                                                     char_model_id,
                                                     concept_id_list)

    def retrieve(self, document_name, query_model_dict):

        bofhmm_ws = Wordspotting(document_name, self.__config)
        patch_def = bofhmm_ws.get_patch_definitions()
        query_patch_dict = patch_def.generate_document_patch_definitions(query_model_dict)
        bofhmm_ws._retrieve_patches(query_patch_dict, query_model_dict,  # IGNORE:protected-access
                                    self.__mm_concept_reader,
                                    self.__char_model_id)


    def query(self, document_name, query_model_dict):
        gw_ws = Wordspotting(document_name, self.__config)
        for querymodel_id, querymodel_def in sorted(query_model_dict.items()):
            gw_ws.process_query(querymodel_id, querymodel_def,
                                False, True, True, True,
                                self.__mm_concept_reader,
                                self.__char_model_id)

class QueryByStringXeval(object):

    def __init__(self, qbs):

        self.__qbs = qbs

    @staticmethod
    def xeval_def(document_list, val_chunk_size):
        """Returns xeval_dict with training-test splits for all cross validation
            folds
        Params:
            document_list: List of documents that the cross validation will be
                performed on.
            val_chunk_size: Size of a single cross validation fold
        Returns:
            xeval_dict: Dictionary with fold identifier as key and tuple of
                training document list and test document list as value.
        """
        logger = logging.getLogger('QueryByStringXeval::xeval_def')
        train_chunk_size = len(document_list) - val_chunk_size
        chunk_size = min(train_chunk_size, val_chunk_size)
        n_partition = int(len(document_list) / chunk_size)
        if n_partition * chunk_size != len(document_list):
            raise ValueError('document_list length must be multiple of '
                             'smaller chunk size (validation chunk size '
                             'or train chunk size)')

        xeval_dict = {}
        logger.info('CROSS VALIDATION SCHEME')
        for i in range(n_partition):
            list_chunk = document_list[chunk_size * i:chunk_size * (i + 1)]
            list_rest = [document for document in document_list
                         if document not in list_chunk]
            xval_id = 'xval-%d-%d-%d' % (train_chunk_size, val_chunk_size, i)
            if chunk_size == val_chunk_size:
                train_list = list_rest
                test_list = list_chunk
            elif chunk_size == train_chunk_size:
                train_list = list_chunk
                test_list = list_rest

            xeval_dict[xval_id] = (train_list, test_list)
            logger.info(' %s', xval_id)
            logger.info('   - training: %s ', ', '.join(train_list))
            logger.info('   - testing : %s \n', ', '.join(test_list))

        return xeval_dict


    def process(self, document_list, val_chunk_size,
                      init=True, retrieve=True, evaluate=True):
        xeval_dict = self.xeval_def(document_list, val_chunk_size)
        if init:
            self.training(xeval_dict)
        if retrieve:
            self.testing(xeval_dict)
        if evaluate:
            self.evaluation(xeval_dict)

    def training(self, xeval_dict):
        logger = logging.getLogger('QueryByStringXeval::training')
        for char_model_id, (train_list, test_list) in sorted(xeval_dict.items()):
            logger.info('TRAINING CHAR MODELS: %s \n', char_model_id)
            self.__qbs.training(train_list, char_model_id)
            logger.info('WRITING RETRIEVAL CONCEPT DEFINITIONS')
            self.__qbs.retrieval_concept_definitions(test_list, char_model_id)

    def testing(self, xeval_dict):
        ret_def_list = self.retrieval_def_list(xeval_dict)
        self.retrieve(ret_def_list, xeval_dict)


    def retrieval_def_list(self, xeval_dict):
        """Create list of document <--> validation split relatations

        Params:
            xeval_dict: Dictionary of {validation_id : (train_list, test_list)}
                where: validation_id is the unique id of the validation split,
                       train_list is the list of document names for training
                       test_list is the list of document names for testing,
                       i.e., retrieval.
        Returns:
            retrieval_def_list: List of
                [ (document_name, validation_id, model_name), ..]
                where model_name is the identifier for the character models
                that where estimated during training, see config.model_name
        """
        retrieval_def_list = []
        for cm_id, (_, test_list) in sorted(xeval_dict.items()):
            retrieval_def_list += self.__qbs.retrieval_def_list(test_list,
                                                                cm_id)
        return retrieval_def_list

    def retrieve(self, retrieval_def_list, xeval_dict):
        logger = logging.getLogger('QueryByStringXeval::retrieve')
        for index, retrieval_def in enumerate(retrieval_def_list):
            char_model_def = retrieval_def[1:]
            # [::-1] get the items reversed: start at index 1 and
            # step backwards (-1)
            dbg_spec = (index, len(retrieval_def_list)) + char_model_def[::-1]
            logger.info('[ %d / %d ] PROCESSING %s: %s \n', *dbg_spec)
            test_list = xeval_dict[char_model_def[0]][1]
            self.__qbs.retrieve(test_list, retrieval_def)

    def evaluation(self, xeval_dict):
        logger = logging.getLogger('QueryByStringXeval::evaluation')
        logger.info('-')
        logger.info('CROSS VALIDATION RESULTS \n')
        n_partition = len(xeval_dict.keys())
        eval_mat = np.zeros((n_partition, 10))
        mip_mat = np.zeros((n_partition, 11))
        for index, cm_id in enumerate(sorted(xeval_dict.keys())):
            test_list = xeval_dict[cm_id][1]
            # obtain partial single value (sv) and multi value (mv) results
            eval_p_sv, eval_p_mv = self.__qbs.evaluation(test_list, cm_id)
            # n_queries, list_m_ap, m_r, m_od, m_ap, m_iap,
            # n_ret_patches, m_ret_time, m_pm_ret_time
            eval_mat[index, :] = eval_p_sv
            mip_mat[index, :] = eval_p_mv

        logger.info('\n%s', str(eval_mat))
        eval_result = eval_mat[:, 1:]
        xeval_weights = eval_mat[:, 0] / np.sum(eval_mat[:, 0])
        eval_result *= np.array([xeval_weights]).T
        xeval_result = np.sum(eval_result, axis=0)
        # xeval_result:
        #
        #         0,   1,    2,    3,     4,
        # list_m_ap, m_r, m_od, m_ap, m_iap,
        #
        #             5,                6,          7,             8
        # n_ret_patches, n_pm_ret_patches, m_ret_time, m_pm_ret_time
        #

        mip_xeval_result = mip_mat * np.array([xeval_weights]).T
        mip_xeval_result = np.sum(mip_xeval_result, axis=0)

        logger.info('mean results:\n%s', str(xeval_result))
        logger.info('-')
        logger.info('Mean Average Precision: %.4f', xeval_result[3])
        logger.info('Mean Recall: \t\t %.4f', xeval_result[1])
        logger.info('Mean Interpolated Average Precision: %.4f',
                    xeval_result[4])
        logger.info('Mean Interpolated Precision:\n%s', str(mip_xeval_result))
        logger.info('-')
        logger.info('Mean Retrieval patches %g', xeval_result[5])
        logger.info('          per model    %g', xeval_result[6])
        logger.info('Mean Retrieval time    %.4f', xeval_result[7])
        logger.info('          per model    %.4f', xeval_result[8])


class QueryByString(object):

    QUERIES_CONCEPT_ID = '.queries'

    def __init__(self, qbs_config):
        self.__config = qbs_config

        self.__char_model_name = qbs_config.model_name
        self.__context_dependent = qbs_config.model_context_dep

        self.__model_path = qbs_config.get_document_model_path(document_id=self.__char_model_name)

        self.__gt_reader = GroundTruthReader(qbs_config.data_base,
                                             gtp_encoding=qbs_config.gtp_encoding)

    def __cm_size_dict_filepath(self, char_model_id, state_index):
        path_params = (self.__model_path, char_model_id, state_index)
        return  '%s%s-msi%d.cm_size_dict.p' % path_params

    def cm_size_dict(self, char_model_id):
        state_index = self.__config.model_state_index
        cm_size_dict_filepath = self.__cm_size_dict_filepath(char_model_id,
                                                             state_index)
        with open(cm_size_dict_filepath, 'rb') as cm_size_fp:
            cm_size_dict = pickle.load(cm_size_fp)
            return cm_size_dict



    def training(self, train_document_list, char_model_id):
        logger = logging.getLogger('QueryByString::training')
        qbs_train = QueryByStringTraining(self.__config, char_model_id)
        cm_size_dict_list = qbs_train.training(train_document_list)
        #
        # Save character model size dictionaries
        #
        logger.info('storing size estimates')
        for msi, cm_size_dict in enumerate(cm_size_dict_list):
            logger.info(' ... model state index %d', msi)
            cm_size_dict_fp = self.__cm_size_dict_filepath(char_model_id,
                                                           state_index=msi)
            with open(cm_size_dict_fp, 'wb') as cmsize_handle:
                pickle.dump(cm_size_dict, cmsize_handle)


    def retrieval_concept_definitions(self, test_document_list, char_model_id):
        # ATTENTION: write concept definitions before initializing MMCharModelReader
        # --> see QueryByStringRetrieval
        query_word_list = self.__query_word_list(test_document_list)
        self.query_concept_definitions(query_word_list, char_model_id)

    def retrieval_def_list(self, test_document_list, char_model_id):
        retrieval_def_list = [(document_name,
                               char_model_id,
                               self.__char_model_name)
                              for document_name in test_document_list]
        return retrieval_def_list


    def retrieve(self, test_document_list, retrieval_def):
        document_name, char_model_id, _ = retrieval_def
        query_word_list = self.__query_word_list(test_document_list)
        self.retrieve_queries(document_name, query_word_list, char_model_id)

    def retrieve_queries(self, document_name, query_word_list, char_model_id):
        cm_size_dict = self.cm_size_dict(char_model_id)
        query_model_dict = self.__query_definitions(query_word_list,
                                                    char_model_id,
                                                    cm_size_dict)
        qbs_test = QueryByStringRetrieval(self.__config, char_model_id)
        qbs_test.retrieve(document_name, query_model_dict)

    def query(self, document_name, query_word_list, char_model_id):
        self.query_concept_definitions(query_word_list, char_model_id)

        qbs_test = QueryByStringRetrieval(self.__config, char_model_id)

        cm_size_dict = self.cm_size_dict(char_model_id)
        query_model_dict = self.__query_definitions(query_word_list,
                                                    char_model_id,
                                                    cm_size_dict)
        qbs_test.query(document_name, query_model_dict)



    def evaluation(self, test_document_list, char_model_id):
        query_word_list = self.__query_word_list(test_document_list)
        return self.evaluation_queries(query_word_list,
                                       test_document_list,
                                       char_model_id)

    def evaluation_queries(self, query_word_list, test_document_list,
                           char_model_id):
        cm_size_dict = self.cm_size_dict(char_model_id)
        query_model_dict = self.__query_definitions(query_word_list,
                                                  char_model_id,
                                                  cm_size_dict)
        model_def = ModelDefinitions(self.__gt_reader)
        gt_model_dict = model_def.generate_document_model_definitions(test_document_list)
        query_results = QueryResultsEvaluator(self.__config, test_document_list,
                                              char_model_id)
        q_res_tup = query_results.evaluate(query_model_dict=query_model_dict,
                                           gt_model_dict=gt_model_dict)
        list_m_ap, m_r, m_od, m_ap, m_iap, m_ip = q_res_tup

        m_r_res_tup = query_results.evaluate_retrieval_time()
        m_ret_patches, m_pm_ret_patches, m_ret_time, m_pm_ret_time = m_r_res_tup
        n_queries = len(query_model_dict)
        # eval single value result tuple
        # multiple value results will be returned separately
        eval_sv_tup = (n_queries, list_m_ap, m_r, m_od, m_ap, m_iap,
                       m_ret_patches, m_pm_ret_patches,
                       m_ret_time, m_pm_ret_time)

        # Delete partial results
        query_id_list = sorted(query_model_dict.keys())
        query_results.delete_partial_retieval_results(query_id_list)
        query_results.delete_partial_timing_results()

        # m_ip is a multi value result and therefore returned separately
        # (see above)
        return eval_sv_tup, m_ip


    def __query_word_list(self, query_document_list):
        #
        # Obtain possible query words
        #
        word_list = []
        for document_id in query_document_list:
            gt_list = self.__gt_reader.read_ground_truth(document_id)
            word_list += [gt_tup[0] for gt_tup in gt_list]
        query_set = set(word_list)
        query_list = sorted(query_set)
        return query_list


    def __query_modellist_dict(self, query_word_list, char_model_list):

        logger = logging.getLogger('QueryByString::__query_modellist_dict')
        #
        # Generate character model lists for each query word
        #
        logger.info('Generating query definitions for decoding')

        charmodel_gen = CharModelGenerator(self.__context_dependent)
        query_modellist_dict = charmodel_gen.transc_modellist_dict(query_word_list)

        if self.__context_dependent:
            #
            # Replace unavailable context dependent models with context
            # independent models
            #
            for model_list in query_modellist_dict.values():
                for cm_idx, cm in enumerate(model_list):
                    if cm not in char_model_list:
                        ci_cm = CharModelGenerator.context_model_to_model(cm)
                        model_list[cm_idx] = ci_cm

        #
        # Filter query_modellist for queries that cannot be decoded with the
        # character models that are available
        #
        for query_word, query_modellist in query_modellist_dict.items():
            if any(cm not in char_model_list for cm in query_modellist):
                del query_modellist_dict[query_word]
                logger.warn('Unsupported query %s', query_word)

        query_modellist_str = ['{:15}   {}'.format(query, ' '.join(models))
                               for query, models in sorted(query_modellist_dict.items())]
        query_modellist_str = '\n'.join(query_modellist_str)
        logger.info('Generated %d queries for retrieval:\n%s',
                    len(query_modellist_dict), query_modellist_str)

        return query_modellist_dict

    def __query_definitions(self, query_word_list, char_model_id, cm_size_dict):
        query_modellist_dict = self.__query_modellist_dict(query_word_list,
                                                           cm_size_dict.keys())
        query_spec_list = self.__query_spec_list(query_modellist_dict,
                                                 cm_size_dict)
        query_model_dict = self.__query_model_dict(query_spec_list,
                                                   char_model_id)
        return query_model_dict


    def query_concept_definitions(self, query_word_list, char_model_id):
        logger = logging.getLogger('QueryByString::query_concept_definitions')
        logger.info('Writing concept definitions for character models: %s',
                    char_model_id)
        cm_size_dict = self.cm_size_dict(char_model_id)
        query_modellist_dict = self.__query_modellist_dict(query_word_list,
                                                           cm_size_dict.keys())
        mm_concept = MMConceptDef(self.__model_path, model_basename=char_model_id)
        queries_concept_id = QueryByString.QUERIES_CONCEPT_ID
        concept_list = []
        for query_name, query_modellist in query_modellist_dict.items():
            concept_def = '%s := \t %s ;' % (query_name, ' '.join(query_modellist))
            concept_list.append(concept_def)

        concept_path = mm_concept.write_decoding_concept(queries_concept_id,
                                                         concept_list)
        logger.info('...wrote queries concept to %s', concept_path)

    @staticmethod
    def __query_spec_list(query_modellist_dict, charmodel_size_dict):
        query_spec_list = []
        for query_name, query_modellist in query_modellist_dict.items():
            query_width = 0
            query_cm_heights = []
            for cm_name in query_modellist:
                cm_size = charmodel_size_dict[cm_name]
                query_width += cm_size[0]
                query_cm_heights.append(cm_size[1])
            query_height = max(query_cm_heights)
            query_size = ((0, 0), (query_width, query_height))
            query_spec_list.append((query_name, query_size))

        return query_spec_list

    @staticmethod
    def __query_model_dict(query_spec_list, char_model_id):
        query_model_dict = {}
        for query_name, query_size in query_spec_list:
            querymodel_id = ModelDefinitions.get_model_id(char_model_id, query_name, 0)
            querymodel_def = (char_model_id, query_name, query_size)
            query_model_dict[querymodel_id] = querymodel_def
        return query_model_dict


class CharModelInitializer(object):

    def __init__(self, config, char_model_id):
        self.__config = config

        gt_reader = GroundTruthReader(base_path=config.data_base,
                                      gtp_encoding=config.gtp_encoding)
        self.__model_def = ModelDefinitions(gt_reader)
        self.__model_topology = config.model_topology
        self.__vw_offset = self.__config.vw_offset
        self.__vw_grid_spacing = self.__config.vw_grid_spacing
        self.__patch_shift_denom = self.__config.patch_shift_denom
        self.__model_patch_size_reduction = self.__config.model_patch_size_reduction
        self.__model_state_index = config.model_state_index
        self.__model_state_num = config.model_state_num
        self.__height_percentile = config.model_height_percentile
        self.__vt_beamwidth = config.vt_beamwidth
        quantization_path = self.__config.get_quantization_path()
        self.__vw_reader = VisualWordsIO(quantization_path)
        self.__tv_seq_gen = SlidingProbTermVector(config.vocabulary_size,
                                                  config.frame_size,
                                                  config.frame_step)
        if config.has_om_model():
            self.__patch_gen_class = PatchTermOMTransformGen
        else:
            self.__patch_gen_class = PatchTermGen
        self.__model_size = self.__patch_gen_class.model_size(config)

        self.__bg_params = Wordspotting.load_bg_model(config)

        self.__context_dependent = config.model_context_dep

        self.__char_model_name = config.model_name
        self.__char_model_id = char_model_id
        self.__model_path = config.get_document_model_path(self.__char_model_name)
        if not exists(self.__model_path):
            makedirs(self.__model_path)

    def __write_model_svq(self, document_models_dict):
        logger = logging.getLogger('CharModelInitializer::__write_model_svq')
        document_list = sorted(document_models_dict.keys())
        model_dict = self.__model_def.generate_document_model_definitions(document_list)
        logger.info('writing SVQ feature files')
        svq_meta_dict = {}
        for document_name in document_list:
            xyvw_array = self.__vw_reader.read_visualwords(document_name,
                                                           self.__vw_offset)

            patch_gen = self.__patch_gen_class.load(self.__config, xyvw_array)
            patch_seq = PatchSequenceGenerator(patch_gen.document_bounds())

            vw_grid_bounds = patch_seq.get_vw_grid_bounds()
            vw_grid_ul = vw_grid_bounds[0]
            patch_quantizer = PatchQuantizer(self.__vw_grid_spacing, vw_grid_ul,
                                             self.__patch_shift_denom,
                                             self.__model_patch_size_reduction)

            svq_path = self.__config.get_svq_path(document_name)
            if not exists(svq_path):
                makedirs(svq_path)
            model_svq_gen = ModelSVQGenerator(patch_gen, patch_quantizer,
                                              self.__model_path, svq_path)
            model_ids = document_models_dict[document_name]
            svqgen_model_dict = {m_id: model_dict[m_id] for m_id in model_ids}
            svq_meta_dict.update(model_svq_gen.write_model_svq(svqgen_model_dict))

        return svq_meta_dict

    def align_character_models(self, document_models_dict):
        logger = logging.getLogger('CharModelInitializer::align_character_models')
        logger.info('initializing')
        document_list = sorted(document_models_dict.keys())
        model_dict = self.__model_def.generate_document_model_definitions(document_list)
        svq_meta_dict = self.__write_model_svq(document_models_dict)
        msi = self.__model_state_index
        mm_base = MMBase(self.__model_path, self.__char_model_id)
        model_state_path = mm_base.get_model_state_path(msi)
        model_definition_path = mm_base.get_model_definition_path(msi)
        concept_path_list = []
        if self.__context_dependent:
            cd_concept_id = CharModelEstimator.CONTEXT_DEP_CONCEPT_ID
            cd_concept_path = mm_base.get_model_concept_path(cd_concept_id)
            concept_path_list.append(cd_concept_path)
        logger.info('loading model')
        mm_definition = MMDefinition(model_state_path,
                                     model_definition_path,
                                     concept_path_list)
        charmodel_gen = CharModelGenerator(self.__context_dependent)
        transc_list = [m_def[1] for m_def in model_dict.values()]
        transc_cmlist_dict = charmodel_gen.transc_modellist_dict(transc_list)
        charmodel_align = CharModelAlignment(model_dict,
                                             svq_meta_dict,
                                             transc_cmlist_dict,
                                             mm_definition,
                                             self.__vt_beamwidth)
        logger.info('forced alignment')
        charmodel_def_list = []
        for m_id in sorted(model_dict.keys()):
            m_cm_def_list = charmodel_align.charmodel_def_list(m_id)
            cm_model_list = [cm_def[1] for cm_def in m_cm_def_list]
            cm_transc_list = charmodel_gen.models_to_transc(cm_model_list)
            m_cm_def_list = [(cm_def[0], cm_t, cm_def[2])
                             for cm_def, cm_t in zip(m_cm_def_list,
                                                     cm_transc_list)]
            charmodel_def_list.extend(m_cm_def_list)

        return charmodel_def_list

    def estimate_character_models(self, document_models_dict):
        logger = logging.getLogger('CharModelInitializer::estimate_character_models')
        document_list = sorted(document_models_dict.keys())
        model_dict = self.__model_def.generate_document_model_definitions(document_list)

        #
        # Write SVQ representations
        #
        svq_meta_dict = self.__write_model_svq(document_models_dict)

        #
        # Estimate character models from word annotations
        #
        logger.info('estimate character models')
        model_ids = list(itertools.chain(*document_models_dict.values()))

        # Always perform white space padding, padded ws models are optional
        # white space model estimation is organized below
        # (based on padding, voting or line-based space estimates)
        whitespace_padding = True
        charmodel_estimator = CharModelEstimator(self.__model_path,
                                                 self.__model_topology,
                                                 self.__model_state_index,
                                                 self.__model_state_num,
                                                 self.__char_model_id,
                                                 self.__model_size,
                                                 self.__context_dependent,
                                                 whitespace_padding)
        word_model_def = charmodel_estimator.estimate_models(model_ids,
                                                             model_dict,
                                                             svq_meta_dict)
        word_charmodellist_dict, charmodel_list, concept_paths = word_model_def

        #
        # Estimate and append additional models
        #
        mminit = MMInit(self.__model_path, model_basename=self.__char_model_id)

        if self.__config.has_ws_estimation_voting():
            logger.info('estimating query whitespace models by voting')
            ws_msi_param_list = self.__whitespace_voting(document_list,
                                                         model_dict)
        elif self.__config.has_ws_estimation_padding():
            logger.info('using padded whitespace models '
                        'as query whitespace models')
            cm_sp_tup = CharModelGenerator.spacepad_leftright_symbols()
            cm_space_left, cm_space_right = cm_sp_tup
            m_name_left, m_name_right = ConceptDefinitions.WS_MODEL_IDS
            cm_src_dst_list = [(cm_space_left, m_name_left),
                               (cm_space_right, m_name_right)]
            ws_msi_param_list = self.__whitespace_load_cm_space(concept_paths,
                                                                cm_src_dst_list)
        elif self.__config.has_ws_estimation_line():
            logger.info('using <space> whitespace model (from line annotation) '
                        'as query whitespace (left | right) model')
            char_gen = CharModelGenerator(context_dependent=False)
            cm_space_id = char_gen.transc_to_models(transcription=' ')
            m_name_left, m_name_right = ConceptDefinitions.WS_MODEL_IDS
            cm_src_dst_list = [(cm_space_id, m_name_left),
                               (cm_space_id, m_name_right)]
            ws_msi_param_list = self.__whitespace_load_cm_space(concept_paths,
                                                                cm_src_dst_list)
        else:
            raise ValueError('No whitespace estimation model has been set ',
                             '(has to be in {vote|pad|line}')

        logger.info('appending additional whitespace models')
        for msi in range(0, self.__model_state_index + 1):
            logger.info(' ... model state index %d', msi)
            ws_params = ws_msi_param_list[msi]
            # Append white space models
            logger.info('     white space models: %s', ws_params.keys())
            mminit.append_model_definition(ws_params,
                                           state_index=msi)

        logger.info('appending additional background model')
        for msi in range(0, self.__model_state_index + 1):
            logger.info(' ... model state index %d', msi)
            # Append background model
            logger.info('     background model(s): %s', self.__bg_params.keys())
            mminit.append_model_definition(self.__bg_params,
                                           state_index=msi)

        #
        # Character model size estimates based on forced alignment
        #
        # this does not include size estimates for white space and background
        # models


        model_size_estimator = CharModelSizeEstimator(model_ids, model_dict,
                                                      svq_meta_dict,
                                                      self.__context_dependent,
                                                      word_charmodellist_dict,
                                                      charmodel_list,
                                                      self.__vw_grid_spacing[0],
                                                      self.__height_percentile,
                                                      self.__vt_beamwidth)

        logger.info('estimate character model size')
        mm_base = MMBase(self.__model_path, self.__char_model_id)
        cm_size_dict_list = []
        for msi in range(0, self.__model_state_index + 1):
            logger.info(' ... model state index %d', msi)
            model_state_path = mm_base.get_model_state_path(msi)
            model_definition_path = mm_base.get_model_definition_path(msi)
            mm_definition = MMDefinition(model_state_path,
                                         model_definition_path,
                                         concept_paths)
            cm_size_dict = model_size_estimator.estimate(mm_definition)
            cm_size_dict_list.append(cm_size_dict)

        return cm_size_dict_list

    def __whitespace_voting(self, document_list, model_dict):
        logger = logging.getLogger('CharModelInitializer::__whitespace_voting')
        # Estimate white space models
        logger.info('   estimating white space models')
        # Collect region_stats from word annotations
        region_hyps = RegionHypotheses()
        for document_name in document_list:
            m_def_list = [m_def for m_def in model_dict.values()
                          if m_def[0] == document_name]
            # region tuple mimics ccspace region, bounding box must be at tuple
            # index 1
            region_list = [(None, m_def[2], None) for m_def in m_def_list ]
            region_hyps.extend_region_list(document_name, region_list)
        # Estimate white space model from word-level annotations
        #
        # Multiple height white space models are unsupported.
        # Using single left-side and right-side models for all
        # white space annotations
        ws_models = WhitespaceModels(self.__config,
                                     context_id=self.__char_model_name,
                                     mm_model_id_prefix=self.__char_model_id)
        ws_models.process_document_list(document_list, region_hyps)

        msi_param_list = []
        for msi in range(0, self.__model_state_index + 1):
            ws_params = ws_models.load_ws_params(m_state_index=msi)
            msi_param_list.append(ws_params)

        return msi_param_list

    def __whitespace_load_cm_space(self, concept_paths, cm_src_dst_list):
        logger = logging.getLogger('CharModelInitializer::'
                                   '__whitespace_load_cm_space')
        mm_base = MMBase(self.__model_path, self.__char_model_id)
        msi_param_list = []
        for msi in range(0, self.__model_state_index + 1):
            logger.info(' ... model state index %d', msi)
            model_state_path = mm_base.get_model_state_path(msi)
            model_definition_path = mm_base.get_model_definition_path(msi)
            # Load model definitions
            mm_definition = MMDefinition(model_state_path,
                                         model_definition_path,
                                         concept_paths)
            # Export model definitions for each source char model id
            ws_params_dict = {}
            for cm_src_id, cm_dst_id in cm_src_dst_list:
                # Define concept for source char model
                ws_concept_def = '%s ;' % cm_src_id
                ws_concept = MMConcept(mm_definition, ws_concept_def)
                # Define data structures for exporting model parameters
                n_transitions = self.__config.get_n_transitions()
                n_states = ws_concept.n_states()
                model_size = self.__config.model_size()
                model_topology = self.__config.model_topology
                transitions_arr = np.zeros((n_states, n_transitions))
                mixtures_arr = np.zeros((n_states, model_size))
                ws_concept.export_state_space(transitions_arr, mixtures_arr)
                # Store model parameters from source char model as destination
                # char model
                ws_params = (transitions_arr, mixtures_arr, model_topology)
                ws_params_dict[cm_dst_id] = ws_params
            msi_param_list.append(ws_params_dict)

        return msi_param_list

    def to_document_models_dict(self, model_spec_list):
        document_models_dict = defaultdict(list)
        for model_spec in model_spec_list:
            m_id = self.__model_def.get_model_id(*model_spec)
            document_models_dict[model_spec[0]].append(m_id)

        return document_models_dict

class CharModelAlignment(object):

    def __init__(self, datum_dict, svq_meta_dict, transc_charmodellist_dict,
                 mm_definition, vt_beamwidth):

        self.__datum_dict = datum_dict
        self.__svq_meta_dict = svq_meta_dict
        self.__transc_charmodellist_dict = transc_charmodellist_dict
        self.__mm_definition = mm_definition
        self.__vt_beamwidth = vt_beamwidth

        self.__hyp_chain = MMHypChain()
        self.__svq_termvectors = SVQVectors()
        self.__svq_termvectors_io = SVQVectorsIO()

    def align_forced(self, datum_id):
        d_transc = self.__datum_dict[datum_id][1]
        d_charmodellist = self.__transc_charmodellist_dict[d_transc]
        concept_def_str = ' '.join(d_charmodellist) + ' ;'
        mm_concept = MMConcept(self.__mm_definition, concept_def_str)


        #
        # Load svq termvectors
        #
        d_svq_filepath = self.__svq_meta_dict[datum_id][0]
        self.__svq_termvectors.reset()
        self.__svq_termvectors_io.read_vectors(d_svq_filepath,
                                               self.__svq_termvectors)

        #
        # Align termvectors with concept
        #
        self.__hyp_chain.clear()
        mm_align = MMAlign(mm_concept, self.__vt_beamwidth)
        mm_align.viterbi_decoding(self.__svq_termvectors, self.__hyp_chain)
        hyp_model_list = self.__hyp_chain.modelname_list()
        hyp_align_list = self.__hyp_chain.modelalign_list()

        if len(hyp_model_list) != len(hyp_align_list):
            raise ValueError('Number of aligned models and segments mismatch!')

        return hyp_model_list, hyp_align_list

    def charmodel_def_list(self, datum_id):
        hyp_cm_list, hyp_align_list = self.align_forced(datum_id)

        d_def = self.__datum_dict[datum_id]
        d_ul_xy = d_def[2][0]
        d_lr_xy = d_def[2][1]

        d_svq_meta = self.__svq_meta_dict[datum_id]
        d_x_idx_tup = d_svq_meta[3]

        cm_def_list = []
        for cm_id, (seg_s_idx, seg_e_idx) in zip(hyp_cm_list, hyp_align_list):
            cm_ul_x = d_x_idx_tup[seg_s_idx]
            cm_lr_x = d_x_idx_tup[seg_e_idx]
            cm_bnds = ((cm_ul_x, d_ul_xy[1]), (cm_lr_x, d_lr_xy[1]))
            cm_def = (d_def[0], cm_id, cm_bnds)
            cm_def_list.append(cm_def)

        return cm_def_list

class CharModelSizeEstimator(object):

    def __init__(self, model_ids, model_dict, svq_meta_dict, context_dependent,
                 word_charmodellist_dict, charmodel_list,
                 align_width_factor, height_percentile, vt_beamwidth):
        self.__model_ids = model_ids
        self.__model_dict = model_dict
        self.__svq_meta_dict = svq_meta_dict
        self.__context_dependent = context_dependent
        self.__word_charmodellist_dict = word_charmodellist_dict
        self.__charmodel_list = charmodel_list
        self.__align_width_factor = align_width_factor
        self.__height_percentile = height_percentile
        self.__vt_beamwidth = vt_beamwidth

    def estimate(self, mm_definition):
        logger = logging.getLogger('CharModelSizeEstimator::estimate')
        #
        # Align models
        #
        logger.info('forced model alignment')
        model_charlenlist_dict = {}
        align_width_factor = self.__align_width_factor
        cm_align = CharModelAlignment(self.__model_dict,
                                      self.__svq_meta_dict,
                                      self.__word_charmodellist_dict,
                                      mm_definition,
                                      self.__vt_beamwidth)
        for m_id in self.__model_ids:
            #
            # Forced alignment
            #
            _, hyp_align_list = cm_align.align_forced(m_id)

            hyp_alignwidth_list = [(h_align[1] + 1 - h_align[0]) *
                                   align_width_factor
                                    for h_align in hyp_align_list]
            model_charlenlist_dict[m_id] = hyp_alignwidth_list


        logger.info('size estimates for character models')
        charmodel_size_dict = self.__charmodel_size_dict(self.__model_ids,
                                                         self.__model_dict,
                                                         model_charlenlist_dict,
                                                         self.__word_charmodellist_dict)
        if self.__context_dependent:
            #
            # Extend charmodel_size_dict with character models that have not
            # been aligned in forced alignment
            #
            # When using context dependent models, the context independent
            # models will not have been aligned, before.
            #
            # Size estimates for the context independent models are obtained
            # as the average of the size estimates of the context dependent
            # ones.
            #
            logger.info('updating size estimates for '
                        'context independent models')
            ci_charmodel_size_dict = defaultdict(list)
            for cd_cm, cd_cm_size in charmodel_size_dict.items():
                ci_cm = CharModelGenerator.context_model_to_model(cd_cm)
                ci_charmodel_size_dict[ci_cm].append(list(cd_cm_size))
                if ci_cm not in self.__charmodel_list:
                    raise ValueError('Size Estimation: Invalid context ' +
                                     'independent character model')
                ci_cm_upd = self.__mean_charmodel_size(ci_charmodel_size_dict,
                                                       self.__height_percentile)
                charmodel_size_dict.update(ci_cm_upd)

            if len(charmodel_size_dict) != len(self.__charmodel_list):
                logger.info('WARNING: pruning models ' +
                            'because no size estimate is available:')
                # charmodel_list contains the complete list of all character
                # models. This includes context dependent models and
                # context independent models, that could have been estimated
                # from training data
                charmodels_pruned = list(set(self.__charmodel_list) -
                                         set(charmodel_size_dict.keys()))
                logger.info(', '.join(charmodels_pruned))

        return charmodel_size_dict

    def __charmodel_size_dict(self, model_ids, model_dict,
                              model_charlenlist_dict,
                              word_charmodellist_dict):
        logger = logging.getLogger('CharModelSizeEstimator::__charmodel_size_dict')
        charmodel_sizes_align_dict = defaultdict(list)
        for m_id in model_ids:
            m_def = model_dict[m_id]

            m_bounds = m_def[2]
            m_height = m_bounds[1][1] - m_bounds[0][1]

            m_charlen_list = model_charlenlist_dict[m_id]

            m_word = m_def[1]
            w_charmodel_list = word_charmodellist_dict[m_word]

            if len(m_charlen_list) == len(w_charmodel_list):
                for charmodel, c_width in zip(w_charmodel_list, m_charlen_list):
                    charmodel_sizes_align_dict[charmodel].append([c_width,
                                                                  m_height])
            else:
                logger.info('skipping model_id %s for character size ' +
                            'estimation -- forced alignment failed.', m_id)

        height_percentile = self.__height_percentile
        charmodel_size_align_dict = self.__mean_charmodel_size(charmodel_sizes_align_dict,
                                                               height_percentile)

        return charmodel_size_align_dict

    @staticmethod
    def __mean_charmodel_size(charmodel_sizes_stat_dict,
                              height_percentile):
        charmodel_size_dict = {}
        for charmodel, charmodel_sizes_list in  charmodel_sizes_stat_dict.items():
            charmodel_sizes = np.array(charmodel_sizes_list, dtype='float64')
            charmodel_width = np.mean(charmodel_sizes[:, 0])
            charmodel_height = np.percentile(charmodel_sizes[:, 1],
                                             height_percentile)
            charmodel_size_dict[charmodel] = (int(charmodel_width),
                                              int(charmodel_height))

        return charmodel_size_dict



