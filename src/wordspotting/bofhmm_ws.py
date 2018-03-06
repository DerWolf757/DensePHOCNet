'''
Created on Nov 18, 2012

@author: leonard
'''

import logging
from os.path import exists
from os import makedirs
import time
import cPickle as pickle
import scipy.spatial.distance
import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt

from bofhwr.wordspotting.gt_reader import GroundTruthReader
from bofhwr.features.visualwords_io import VisualWordsIO
from bofhwr.features.patch_termvector_sequence import PatchTermvectorGenerator as PatchTermGen, PatchQuantizer , \
    DocumentPatchProcessor, PatchSequenceGenerator, LineRegionSequenceGenerator
from bofhwr.features.patch_sp_processor import DocumentPatchFeatureProcessor, \
     PatchSpatialPyramidGenerator
from bofhwr.features.om_termvector_processor import PatchTermvectorOutputModelTransformGenerator as PatchTermOMTransformGen
from bofhwr.wordspotting.definitions import ModelDefinitions, PatchDefinitions
from bofhwr.wordspotting.model import ModelSVQGenerator, ModelEstimator
from bofhwr.wordspotting.retrieval import RetrievalGenerator, PatchRetrievalProcessor, \
    IFSAccuRetrievalProcessor
from bofhwr.wordspotting.visualization import RetrievalVisualization, ScoreVisualization
from bofhwr.hmm.mm_reader import MMWordModelReader, ConceptDefinitions
from bofhwr.wordspotting.query_results import QueryResultsEvaluator

from patrec.serialization.matrix_io import MatrixIO
from bofhwr.wordspotting.inverted_file_structure import InvertedFileStructure, \
    InvertedFrameStructure
from bofhwr.wordspotting.inverted_file_structure import InvertedFileStructureAccu
from scipy.ndimage import binary_dilation
from patrec.evaluation.retrieval import IterativeMean
from patrec.serialization.list_io import LineListIO
from bofhwr.wordspotting.model import  VWModelGenerator
from bofhwr.wordspotting.visualization import GTVisualization
from bofhwr.wordspotting.retrieval import PatchDistanceRetrievalProcessor, \
    RetrievalResults
from bofhwr.segmentation.region_hyp import WSBoFRegionHyp, WSBoFCCSpace

from wis.mser.regions import RegionHypotheses, RegionProcessor, \
    PatchRegionHypotheses
from wis.mser.ccspace import im_label_index
from bofhwr.features.visualwords_io import VisualWordMatGenerator

class Wordspotting(object):
    '''
    Driver class for running BOF-HMM-based word spotting

    '''

    # Low probability bounds for estimating lower score bound for
    # Viterbi decoding path
    __TRANSITION_PROB_LOW = 5e-01

    @staticmethod
    def load_bg_model(config):
        quantization_path = config.get_quantization_path()
        vw_reader = VisualWordsIO(quantization_path)

        if config.has_om_model():
            patch_gen_class = PatchTermOMTransformGen
            model_params = PatchTermOMTransformGen.load_model(config)
            # First model params ndarray contains (global) mixture weights
            bg_distribution = model_params[0]
            bg_distribution = np.array([bg_distribution])
        else:
            patch_gen_class = PatchTermGen
            model_size = patch_gen_class.model_size(config)
            vocabulary_size = config.vocabulary_size
            codebook_apriori = vw_reader.read_codebook_apriori(vocabulary_size)
            if model_size == codebook_apriori.shape[1] + 1:
                bg_distribution = np.zeros((1, model_size), np.float32)
                bg_distribution[0, :model_size - 1] = codebook_apriori
                bg_distribution[0, model_size - 1] = (1.0 / model_size) * 1.0e-03
                bg_distribution /= np.sum(bg_distribution)
            elif model_size == codebook_apriori.shape[1]:
                bg_distribution = codebook_apriori
            else:
                raise ValueError('Invalid model size in bg model creation')


        next_prob = 1e-09
        skip_prob = 1e-09
        if config.get_n_transitions() == 2:
            self_prob = 1 - next_prob
            bg_transition = np.array([[self_prob, next_prob]])
        elif config.get_n_transitions() == 3:
            self_prob = 1 - next_prob - skip_prob
            bg_transition = np.array([[self_prob, next_prob, skip_prob]])
        else:
            raise ValueError('Unsupported number of '
                             'state transition probabilities')

        bg_model_id = ConceptDefinitions.BG_MODEL_ID
        bg_params = {bg_model_id: (bg_transition,
                                   bg_distribution,
                                   config.model_topology)}
        return bg_params


    def __init__(self, document_name, config, init_decoding=True):
        '''
        Constructor

        @param document_name: Name of the document that will be processed.

        '''
        logger = logging.getLogger('Wordspotting::__init__')
        logger.info('[Wordspotting] document: %s', document_name)

        self.__document_name = document_name
        self.__config = config

        data_base = config.data_base
        vw_offset = config.vw_offset
        vw_grid_spacing = config.vw_grid_spacing
        patch_shift_denom = config.patch_shift_denom
        model_patch_size_reduction = config.model_patch_size_reduction

        svq_path = config.get_svq_path(document_name)
        if not exists(svq_path):
            makedirs(svq_path)

        aligntmp_path = config.get_aligntmp_path(document_name)
        if not exists(aligntmp_path):
            makedirs(aligntmp_path)

        quantization_path = config.get_quantization_path()
        if not exists(quantization_path):
            makedirs(quantization_path)

        document_model_path = config.get_document_model_path(document_name)
        if not exists(document_model_path):
            makedirs(document_model_path)

        scores_path = config.get_scores_path(document_name)
        if not exists(scores_path):
            makedirs(scores_path)


        #
        # Prepare basic data structure (needed by member functions)
        #

        self.__gt_reader = GroundTruthReader(data_base,
                                             gtp_encoding=config.gtp_encoding)

        vw_reader = VisualWordsIO(quantization_path)
        xyvw_array = vw_reader.read_visualwords(self.__document_name, vw_offset)


        if config.has_om_model():
            patch_gen_class = PatchTermOMTransformGen
        else:
            patch_gen_class = PatchTermGen

        concept_def = ConceptDefinitions(config)
        self.__n_patch_models = concept_def.n_patch_models()

        bg_params = Wordspotting.load_bg_model(config)

        patch_width_expansion = 0.0
        if config.has_patch_fitting():
            patch_width_expansion = config.patch_fitting[1]
        whitespace_models = None
        # If the region hypotheses are available, white space models can be
        # estimated by default.
        if config.has_patch_hyp_config():
            m_height_def = config.patch_ws_fitting
            m_height_list = range(m_height_def[0], m_height_def[1] + 1,
                                  m_height_def[2])
            whitespace_models = WhitespaceModels(config, m_height_list)

        model_size = patch_gen_class.model_size(config)
        self.__patch_gen = patch_gen_class.load(config, xyvw_array)
        document_bounds = self.__patch_gen.document_bounds()

        model_path = config.get_document_model_path(document_name)

        vw_grid_ul = document_bounds[:2]
        self.__patch_quantizer = PatchQuantizer(vw_grid_spacing, vw_grid_ul,
                                                patch_shift_denom,
                                                model_patch_size_reduction,
                                                patch_width_expansion)
        store_vwmat = False
        if config.use_sp_decoding():
            logger.info('performing initialization for sp decoding')
            store_vwmat = True
        self.__model_svq_gen = ModelSVQGenerator(self.__patch_gen,
                                                 self.__patch_quantizer,
                                                 model_path,
                                                 svq_path,
                                                 store_vwmat=store_vwmat)

        # Attention: Use adjusted vocabulary size --> model_size
        self.__model_estimator = ModelEstimator(model_path, model_size,
                                                config.model_state_index,
                                                config.model_topology,
                                                config.model_frame_state_fun,
                                                bg_params,
                                                whitespace_models)
        self.__model_definitions = ModelDefinitions(self.__gt_reader)
        self.__document_model_dict = self.__model_definitions.generate_document_model_definitions([self.__document_name])

        self.__patch_seq = None
        self.__doc_patch_gen = None
        self.__doc_patch_sp_gen = None
        self.__patch_sp_gen = None
        self.__patch_definitions = None
        self.__vw_model_gen = None
        self.__retrieval_results = None
        self.__retrieval_gen = None
        self.__ifs = None

        if init_decoding:
            self.__init_decoding(xyvw_array, model_size)
        else:
            logger.info('decoding is disabled... finished initialization.')

    def __init_decoding(self, xyvw_array, model_size):

        logger = logging.getLogger('Wordspotting::__init_decoding')
        config = self.__config
        document_name = self.__document_name
        document_bounds = self.__patch_gen.document_bounds()

        logger.info('patch sequence generator')
        if config.has_patch_line_hyp_config():
            logger.info('using line hypotheses')
            cc_space = WSBoFCCSpace(config)
            line_hyp = WSBoFRegionHyp(config, cc_space)
            height_ypos_dict = line_hyp.load_line_hypotheses(document_name)
            vw_grid_spacing = config.vw_grid_spacing
            self.__patch_seq = LineRegionSequenceGenerator(document_bounds,
                                                           height_ypos_dict,
                                                           vw_grid_spacing[1],
                                                           *config.patch_hyp_line[1:])
        else:
            self.__patch_seq = PatchSequenceGenerator(document_bounds)

        if config.has_patch_region_hyp_config():
            logger.info('patch hypotheses fitter')
            cc_space = WSBoFCCSpace(config)
            reg_hyp = WSBoFRegionHyp(config, cc_space)
            region_list = reg_hyp.load_cc_hypotheses(document_name)
            patch_hyp_fitter = PatchRegionHypotheses(region_list,
                                                     config.patch_hyp_region[1:])
        else:
            patch_hyp_fitter = None


        logger.info('patch-based retrieval')
        self.__doc_patch_gen = DocumentPatchProcessor(self.__patch_gen,
                                                      self.__patch_seq)


        single_query_decoding = config.single_query_decoding
        self.__patch_definitions = PatchDefinitions(document_name,
                                                    self.__gt_reader,
                                                    self.__patch_quantizer,
                                                    single_query_decoding)

        self.__vw_model_gen = VWModelGenerator(config, model_size)
        self.__retrieval_results = RetrievalResults(config, document_name)
        self.__retrieval_gen = RetrievalGenerator(self.__retrieval_results,
                                                  self.__patch_quantizer,
                                                  self.__document_model_dict,
                                                  config.patch_page_thresh,
                                                  config.patch_overlap,
                                                  config.patch_nms,
                                                  patch_hyp_fitter)

        if self.config.use_ifs_vw_indexing():
            if config.has_om_model():
                raise ValueError('Inverted File Structure decoding '
                                 'not supported for visual word indexing')
            #
            # Generate inverted file structure
            #
            logger.info('inverted file structure (visual word indexing)')
            self.__ifs = InvertedFileStructure(xyvw_array, model_size)

        elif self.config.use_ifs_frame_indexing():
            if not config.has_patch_line_hyp_config():
                raise ValueError('Inverted File Structure decoding with '
                                 'frame indexing is only supported with '
                                 'line hyps')
            logger.info('inverted file structure (frame indexing)')
            frame_spec = (config.frame_size,
                          config.frame_step,
                          config.frame_dir)
            self.__ifs = InvertedFrameStructure(self.__patch_seq,
                                                self.__doc_patch_gen,
                                                self.__patch_quantizer,
                                                frame_spec,
                                                model_size,
                                                config.use_ifs_frame_caching())


        if config.use_sp_decoding():
            logger.info('spatial pyramid retrieval')
            sp_partitions = config.sp_partitions
            vocabulary_size = config.vocabulary_size
            self.__patch_sp_gen = PatchSpatialPyramidGenerator(self.__patch_gen,
                                                               vocabulary_size,
                                                               sp_partitions)
            self.__doc_patch_sp_gen = DocumentPatchFeatureProcessor(self.__patch_sp_gen)

    def get_model_svq_generator(self):
        return self.__model_svq_gen

    def get_patch_definitions(self):
        return self.__patch_definitions

    def get_patch_quantizer(self):
        return self.__patch_quantizer

    def get_config(self):
        return self.__config

    config = property(get_config, None, None, None)


    def query_init(self, querymodel_id, querymodel_def):
        if querymodel_def[0] != self.__document_name:
            raise ValueError('Cannot initialize model for different document!')

        querymodel_dict = dict([(querymodel_id, querymodel_def)])
        self._build_model(querymodel_dict)

    def query_retrieve(self, querymodel_id, querymodel_def,
                       mm_concept_reader=None,
                       eval_id=None):
        logger = logging.getLogger('Wordspotting::query_retrieve')
        if mm_concept_reader is None:
            mm_concept_reader = MMWordModelReader(self.config)

        querymodel_dict = dict([(querymodel_id, querymodel_def)])
        patch_document = self.__document_name
        querymodel_bounds = querymodel_def[2]
        querymodel_size_snp = self.__patch_quantizer.get_patch_size_snp(querymodel_bounds)
        document_patch_key = self.__patch_definitions.get_svq_list_id(document_id=patch_document, patch_size=querymodel_size_snp)
        document_patch_dict = dict([(document_patch_key , (patch_document, querymodel_size_snp, [querymodel_id]))])

        t_start = time.time()
        self._retrieve_patches(document_patch_dict, querymodel_dict,
                               mm_concept_reader, eval_id)
        t_end = time.time()
        logger.info('[process_query] retrieve_patches took %.6f seconds',
                    t_end - t_start)

        return (self.__retrieval_results.retrieval_mat(),
                self.__retrieval_results.score_mat())

    def process_query(self, querymodel_id, querymodel_def,
                      init, retrieve, evaluate, visualize,
                      mm_concept_reader=None,
                      eval_id=None):
        '''
        '''


        patch_document = self.__document_name

        querymodel_dict = dict([(querymodel_id, querymodel_def)])

        querymodel_word = querymodel_def[1]
        querymodel_bounds = querymodel_def[2]
        querymodel_size_snp = self.__patch_quantizer.get_patch_size_snp(querymodel_bounds)

        if init:
            self.query_init(querymodel_id, querymodel_def)

        if retrieve:
            self.query_retrieve(querymodel_id, querymodel_def,
                                                   mm_concept_reader, eval_id)
        if evaluate:
            ws_eval = QueryResultsEvaluator(self.config, retrieval_document_list=[patch_document])
            # ATTENTION: Do NOT store results from evaluating a single query!
            # They might conflict with results from all queries!
            ws_eval.evaluate(query_model_dict=querymodel_dict, \
                             gt_model_dict=self.__document_model_dict, store_results=False)

        if visualize:
            # Load retrieval_mat
            retrieval_mat_filepath = self.config.get_retrieval_mat_filepath(patch_document, querymodel_id)
            matrixio = MatrixIO()
            retrieval_mat = matrixio.read_matrix_as_byte(retrieval_mat_filepath, dim=6, missing_dim=True, missing_inner_dim=True)
            # Show results
            score_mat = None
            score_mat_bounds = self.get_score_mat_bounds(querymodel_size_snp)
            if self.config.store_score_mat :
                score_mat_filepath = self.__config.get_score_mat_filepath(patch_document, querymodel_id)
                arr_dict = np.load(score_mat_filepath)
                score_mat = arr_dict['score_mat']

            self.visualize_query(querymodel_word, retrieval_mat, score_mat, score_mat_bounds)

    def visualize_query(self, queryword, retrieval_mat, score_mat, score_mat_bounds):


        # Show results
        if retrieval_mat is not None:
            fig_ret = plt.figure()
            retrieval_vis = RetrievalVisualization(self.config, self.__document_name)
            retrieval_vis.visualize_retrieval_mat(retrieval_mat, fig=fig_ret)
            ax = fig_ret.get_axes()[0]
            gt_vis = GTVisualization(self.config)
            gt_vis.visualize_gt(self.__document_name, queryword, ax=ax)

        fig_score = None
        if score_mat is not None:


            fig_score = plt.figure()

            score_mat_vis = ScoreVisualization(self.config)
            score_mat_vis.visualize_score_mat(self.__document_name, score_mat,
                                              score_mat_bounds,
                                              fig=fig_score)

        plt.show()
        plt.close()
        if fig_score is not None:
            plt.close(fig_score)

    def get_score_mat_bounds(self, querymodel_size_snp):
        # Determine score_mat bounds in the document images
        # For each patch a score is obtained
        querymodel_patch_shift_snp = self.__patch_quantizer.get_patch_shift_snp(querymodel_size_snp)

        if self.config.decoding_mode == 'ifs':
            cell_size_snp = querymodel_patch_shift_snp
        else:
            cell_size_snp = querymodel_size_snp

        document_bounds = self.__patch_gen.document_bounds()
        patch_seq = PatchSequenceGenerator(document_bounds)
        p_ul_xy, p_lr_xy = patch_seq.get_patch_bounds(cell_size_snp,
                                                   querymodel_patch_shift_snp)
        # Coordinates in the patch list refer to the upper left patch
        # corner. For an nicer visualization the scores should be shown
        # at the respective patch centers
        # --> shift the bounds by patch_size/2 in x and y direction
        x_min = p_ul_xy[0] + cell_size_snp[0] / 2
        x_max = p_lr_xy[0] + cell_size_snp[0] / 2
        y_min = p_ul_xy[1] + cell_size_snp[1] / 2
        y_max = p_lr_xy[1] + cell_size_snp[1] / 2
        score_mat_bounds = (x_min, y_min, x_max, y_max)

        return score_mat_bounds



    def batch_init(self):
        logger = logging.getLogger('Wordspotting::batch_init')
        logger.info('\n\n###INIT##############################' +
                    '###########################')
        logger.info('Going to initialize models...')
        logger.info('#########################################' +
                    '#######################')

        logger.info('Loading query specifications...')
        document_model_dict = self.__document_model_dict
        logger.info('%d query specifications for initialization',
                     len(document_model_dict))
        self._build_model(document_model_dict)


    def batch_retrieve(self, document_list, eval_id=None):
        '''

        '''
        logger = logging.getLogger('Wordspotting::batch_retrieve')
        logger.info('n#######################################' +
                    '###########################')
        logger.info('WORDSPOTTING::PROCESS_BATCH\n')

        #
        # Prepare document-list query and document patch representations
        #
        logger.info('Loading query specifications...')
        decode_model_dict = self.__model_definitions.generate_document_model_definitions(document_list)
        logger.info('%d query specifications for decoding',
                    len(decode_model_dict))

        decode_patch_dict = self.__patch_definitions.generate_document_patch_definitions(decode_model_dict)


        logger.info('\n')
        logger.info('###RETRIEVE##########################' +
                    '###################')
        logger.info('Going to retrieve patches according to query models...')
        logger.info('#########################################' +
                    '#######################')
        t_start = time.time()
        mm_concept_reader = MMWordModelReader(self.config)
        self._retrieve_patches(decode_patch_dict, decode_model_dict,
                               mm_concept_reader, eval_id)
        t_end = time.time()
        logger.info('Patch retrievel took %.6f seconds', t_end - t_start)


        logger.info('\n\n###FINISHED##########################' +
                    '###########################')



    def _build_model(self, doc_model_dict):
        """Builds the HMM spotting model from the annotations

        Params:
            doc_model_dict: dict of (model_id, model_definitions). For each
                element an HMM model will be estimated.
        """

        # Write SVQ data
        model_svqmeta_dict = self.__model_svq_gen.write_model_svq(doc_model_dict)

        # Generate / write HMM model definitions
        meta_model_dict = {m_id : tuple([m_id]) for m_id in doc_model_dict}
        self.__model_estimator.estimate_model(meta_model_dict, model_svqmeta_dict)


    def _retrieve_patches(self, doc_patch_dict, decode_model_dict,
                          mm_concept_reader, eval_id):
        '''
         Retrieve doc_patch_dict
         1. Generate patch bof representations
         2. Decode the models matching the patch dimensions
         3. Retrieve the top rated patches for the given page
         In an integrated manner where no data is written to disk in between.

        '''
        logger = logging.getLogger('Wordspotting:_retrieve_patches')
        mean_retrieval_patches = IterativeMean()
        mean_model_retrieval_patches = IterativeMean()
        mean_retrieval_time = IterativeMean()
        mean_model_retrieval_time = IterativeMean()

        doc_patch_num = len(doc_patch_dict)
        for index, (doc_patch_key, doc_patch_def) in enumerate(sorted(doc_patch_dict.items())):
            logger.info('[ %04d / %04d ] going to process doc_patch request',
                        index, doc_patch_num)
            #
            # Prepare patch definition
            #
            patch_size_snp = doc_patch_def[1]
            patch_shift_snp = self.__patch_quantizer.get_patch_shift_snp(patch_size_snp)
            logger.info('patch_size_snp ( %d, %d )',
                        patch_size_snp[0],
                        patch_size_snp[1])
            logger.info('patch_shift_snp ( %d, %d )',
                        patch_shift_snp[0],
                        patch_shift_snp[1])
            #
            # Initialize PatchRetrievalProcessor
            #
            # doc_patch_def[2] contains all model ids that will be decoded for
            # (snapped) patch size: doc_patch_def[1].
            #
            model_list = doc_patch_def[2]
            logger.info('Going to load HMM definitions')
            mm_concept_dict = mm_concept_reader.read_mm_concepts(model_list,
                                                                 decode_model_dict)
            mm_concept_name_dict = mm_concept_reader.model_concept_name_dict(model_list,
                                                                             decode_model_dict)
            # Use bg norm causes the query path probability to be isolated from
            # background model path probabilities
            use_bg_norm = self.config.has_bg_normalization()


            if self.config.use_sp_decoding():
                sp_distance = self.config.sp_distance
                sp_res_tup = self.__sp_retrieval(decode_model_dict,
                                                 model_list,
                                                 doc_patch_key,
                                                 patch_size_snp,
                                                 patch_shift_snp,
                                                 sp_distance)
                retrieval_proc, ret_time, n_ret_patches = sp_res_tup
                ret_time += retrieval_proc.retrieve_patches()
                del retrieval_proc
            elif (self.config.use_ifs_decoding() and
                  not self.config.use_vt_decoding()):
                ifs_res_tup = self.__ifs_retrieval(decode_model_dict,
                                                   model_list,
                                                   mm_concept_dict,
                                                   patch_size_snp,
                                                   patch_shift_snp)
                retrieval_proc, ret_time = ifs_res_tup
                # Inverted file structure decoding does require to process
                # individual patches
                # --> number of process patches for retrieval is zero
                n_ret_patches = 0
                ret_time += retrieval_proc.retrieve_patches()
                del retrieval_proc

            elif (not self.config.use_ifs_decoding() and
                  self.config.use_vt_decoding()):
                vt_res_tup = self.__vt_retrieval(decode_model_dict,
                                                 model_list,
                                                 mm_concept_dict,
                                                 use_bg_norm,
                                                 mm_concept_name_dict,
                                                 doc_patch_key,
                                                 patch_size_snp,
                                                 patch_shift_snp)
                retrieval_proc, ret_time, n_ret_patches = vt_res_tup
                ret_time += retrieval_proc.retrieve_patches()
                del retrieval_proc

            elif (self.config.use_ifs_decoding() and
                  self.config.use_vt_decoding()):
                #
                # ATTENTION ifs accu cell size and vt patch shift must match
                # for masking the dense vt patch grid accu cells must be
                # associated with vt patch positions
                #
                ifs_res_tup = self.__ifs_retrieval(decode_model_dict,
                                                   model_list,
                                                   mm_concept_dict,
                                                   patch_size_snp,
                                                   patch_shift_snp)
                ifs_retrieval_proc, ifs_time = ifs_res_tup
                t_start = time.time()
                ifs_retrieval_proc.prune_scores(patch_size_snp)
                if (self.config.vw_accu_filter[0] >= 1 and
                    self.config.vw_accu_filter[1] >= 1):
                    ifs_dilation_filter = self.config.vw_accu_filter
                    dilation_mask = np.ones(ifs_dilation_filter)
                    ifs_retrieval_proc.filter_scores(binary_dilation,
                                                     dilation_mask)
                    ifs_accu_filter = ifs_dilation_filter
                else:
                    ifs_accu_filter = (1, 1)

                model_patchmask_dict = ifs_retrieval_proc.model_scoremat_dict()
                #
                # Shift patch masks to adapt them to vt patch retrieval
                # IFS patches are usually detected at patch centers while
                # vt retrieval is based on upper left patch corners
                # --> Shift masks according to patch_size and pad with zeros
                # ATTENTION: The patch_shift must be equal for ifs and vt
                # retrieval
                #
                # Shift half the snapped patch size
                mask_shift = (np.array(patch_size_snp) /
                              np.array(patch_shift_snp)) / 2
                for m_id in model_list:
                    m_patchmask_mat = model_patchmask_dict[m_id]
                    m_patchmask_mat = m_patchmask_mat[mask_shift[1]:,
                                                      mask_shift[0]:]
                    model_patchmask_dict[m_id] = m_patchmask_mat

                t_end = time.time()
                accu_filter_time = t_end - t_start
                vt_res_tup = self.__vt_retrieval(decode_model_dict,
                                                 model_list,
                                                 mm_concept_dict,
                                                 use_bg_norm,
                                                 mm_concept_name_dict,
                                                 doc_patch_key,
                                                 patch_size_snp,
                                                 patch_shift_snp,
                                                 model_patchmask_dict)
                vt_retrieval_proc, vt_time, n_ret_patches = vt_res_tup
                ret_time = vt_retrieval_proc.retrieve_patches(ifs_accu_filter)
                ret_time += ifs_time + accu_filter_time + vt_time
                del ifs_retrieval_proc
                del vt_retrieval_proc
            else:
                raise ValueError('Unknown decoding mode: %s' %
                                 self.config.decoding_mode)

            n_m_ret_patches = n_ret_patches / float(len(model_list))
            m_ret_time = ret_time / len(model_list)
            logger.info('Retrieval time ( %.6f seconds)', ret_time)
            logger.info('Retrieval time per model ( %.6f seconds)',
                        ret_time / len(model_list))
            mean_retrieval_patches.add_value(n_ret_patches)
            mean_model_retrieval_patches.add_value(n_m_ret_patches)
            mean_retrieval_time.add_value(ret_time)
            mean_model_retrieval_time.add_value(m_ret_time)

        retrieval_time_list = [mean_retrieval_patches.get_mean(),
                               mean_model_retrieval_patches.get_mean(),
                               mean_retrieval_time.get_mean(),
                               mean_model_retrieval_time.get_mean()]
        logger.info('MEAN RETRIEVAL PATCHES: %g', retrieval_time_list[0])
        logger.info('MEAN MODEL RETRIEVAL PATCHES: %g', retrieval_time_list[1])
        logger.info('MEAN RETRIEVAL TIME: %.6f', retrieval_time_list[2])
        logger.info('MEAN MODEL RETRIEVAL TIME: %.6f', retrieval_time_list[3])
        retrieval_time_fp = self.config.get_retrieval_time_filepath(self.__document_name,
                                                                    eval_id)
        LineListIO.write_list(retrieval_time_fp,
                              ['%.6f' % ret_time
                               for ret_time in retrieval_time_list])


    def __vt_retrieval(self, decode_model_dict, model_list,
                       mm_concept_dict, use_bg_norm, mm_concept_name_dict,
                       doc_patch_key, patch_size_snp, patch_shift_snp,
                       model_patchmask_dict=None):
        logger = logging.getLogger('Wordspotting::_vt_retrieval')
        # patch_yx_mat contains all patch coordinates, row-wise
        patch_seq_tup = self.__patch_seq.get_patch_matrix(patch_size_snp,
                                                          patch_shift_snp)
        patch_yx_mat, patch_mat_dim = patch_seq_tup
        if model_patchmask_dict is None:
            patchmask = np.ones(patch_mat_dim)
            model_patchmask_dict = {m_id : patchmask for m_id in model_list}

        # Estimate lower bound for Viterbi patch score
        # neg-log: the larger the number, the lower the probability
        #
        # lower bound for transition probability
        nlog_trans_prob_bound = -np.log(Wordspotting.__TRANSITION_PROB_LOW)
        # lower bound for transition probability
        nlog_output_prob_bound = -np.log(self.config.vt_problow * 1e-01)
        # multiply transition and output probability for a Viterbi decoding
        # path of length n_frames (log-domain)
        patch_score_bound = nlog_trans_prob_bound + nlog_output_prob_bound
        logger.info('Patch score bound: %g', patch_score_bound)
        retrieval_processor = PatchRetrievalProcessor(self.__retrieval_gen,
                                                      model_list,
                                                      decode_model_dict,
                                                      mm_concept_dict,
                                                      use_bg_norm,
                                                      mm_concept_name_dict,
                                                      model_patchmask_dict,
                                                      patch_yx_mat,
                                                      patch_mat_dim,
                                                      self.__patch_quantizer,
                                                      patch_score_bound,
                                                      self.config.vt_beamwidth,
                                                      self.config.vt_problow)
        patchmat_mask = retrieval_processor.patchmask_mat()
        #
        # Generate and process patch representations
        #
        t_start = time.time()
        self.__doc_patch_gen.process_document(doc_patch_key,
                                              retrieval_processor,
                                              patch_mat_dim,
                                              patch_size_snp,
                                              patch_shift_snp,
                                              patchmat_mask)
        t_end = time.time()
        patch_decoding_time = t_end - t_start
        n_patches_decoded = retrieval_processor.n_patches_decoded()
        logger.info('Finished patch decoding ( %d patches in %.6f seconds )',
                    n_patches_decoded,
                    patch_decoding_time)

        return retrieval_processor, patch_decoding_time, n_patches_decoded



    def __ifs_retrieval(self, decode_model_dict, model_list, model_concept_dict,
                        patch_size_snp, patch_shift_snp):
        logger = logging.getLogger('Wordspotting::__ifs_retrieval')
        t_start = time.time()
        model_vw_list = self.__vw_model_gen.generate_hmm_model_vwlist_list(model_list,
                                                                           model_concept_dict,
                                                                           self.__n_patch_models)
        ifs_accu = InvertedFileStructureAccu(self.__ifs, self.__patch_seq,
                                             patch_shift_snp, patch_size_snp)
        retrieval_processor = IFSAccuRetrievalProcessor(self.__retrieval_gen,
                                                        model_list,
                                                        decode_model_dict,
                                                        model_vw_list,
                                                        self.__patch_quantizer,
                                                        ifs_accu)
        t_end = time.time()
        logger.info('IFS accu. initialization took %.6f seconds',
                    t_end - t_start)
        t_start = time.time()
        retrieval_processor.process_ifs_model_accus()

        t_end = time.time()
        accu_decoding_time = t_end - t_start
        logger.info('Finished inverted file structure ' +
                    'accu decoding ( %.6f seconds)', accu_decoding_time)
        return retrieval_processor, accu_decoding_time


    def __sp_retrieval(self, decode_model_dict, model_list, doc_patch_key,
                       patch_size_snp, patch_shift_snp, distance_metric):
        logger = logging.getLogger('Wordspotting::__sp_retrieval')
        # Re-initialize PatchSequencGenerator: Spatial Pyramid mode only
        # supports brute-force patch processing
        document_bounds = self.__patch_gen.document_bounds()
        patch_seq = PatchSequenceGenerator(document_bounds)
        patch_seq_tup = patch_seq.get_patch_matrix(patch_size_snp,
                                                   patch_shift_snp)
        patch_yx_mat, patch_mat_dim = patch_seq_tup
        n_y_pos, n_x_pos = patch_mat_dim
        patch_idx_mat = patch_seq.to_patch_idx_matrix(xrange(n_x_pos),
                                                      xrange(n_y_pos))


        logger.info('Going to load model visual words ' +
                    '(for computing reference Spatial Pyramids)')
        model_vwmat_list = self.__vw_model_gen.load_model_vwmat_list(model_list,
                                                                     decode_model_dict)

        retrieval_processor = PatchDistanceRetrievalProcessor(self.__retrieval_gen,
                                                              model_list,
                                                              decode_model_dict,
                                                              model_vwmat_list,
                                                              self.__patch_sp_gen,
                                                              patch_yx_mat,
                                                              patch_mat_dim,
                                                              self.__patch_quantizer,
                                                              distance_metric)
        #
        # Generate and process patch representations
        #
        t_start = time.time()
        self.__doc_patch_sp_gen.process_document(doc_patch_key,
                                                 retrieval_processor,
                                                 patch_yx_mat,
                                                 patch_idx_mat,
                                                 patch_size_snp)
        t_end = time.time()
        patch_decoding_time = t_end - t_start
        n_patches_proc = n_x_pos * n_y_pos
        logger.info('Finished patch decoding ( %d patches in %.6f seconds )',
                    n_patches_proc,
                    patch_decoding_time)

        return retrieval_processor, patch_decoding_time, n_patches_proc


class MetaModelInitializer(object):


    __META_CONTEXT_ID = 'meta'

    def __init__(self, config, document_list=None, doc_model_dict=None,
                 context_id=__META_CONTEXT_ID, cleanup_training=True):

        self.__config = config
        self.__context_id = context_id
        self.__cleanup_training = cleanup_training
        if 'sp' in config.decoding_mode:
            raise ValueError('Spatial Pyramid decoding for Meta models is unsupported!')

        if document_list is not None:
            gt_reader = GroundTruthReader(config.data_base,
                                          gtp_encoding=config.gtp_encoding)
            model_def = ModelDefinitions(gt_reader)
            self.__doclist_model_dict = model_def.generate_document_model_definitions(document_list)
        elif doc_model_dict is not None:
            self.__doclist_model_dict = doc_model_dict
        else:
            raise ValueError('document_list or doc_model_dict must be provided')

        model_path = config.get_document_model_path(context_id)
        if not exists(model_path):
            makedirs(model_path)

        if config.has_om_model():
            self.__patch_gen_class = PatchTermOMTransformGen
        else:
            self.__patch_gen_class = PatchTermGen
        model_size = self.__patch_gen_class.model_size(config)
        self.__model_estimator = ModelEstimator(model_path, model_size,
                                                config.model_state_index,
                                                config.model_topology,
                                                config.model_frame_state_fun)

    def init_models(self, meta_model_dict):

        meta_model_def_dict = {}
        for meta_model_name, model_id_list in sorted(meta_model_dict.items()):
            mm_id, mm_def = self.init(meta_model_name, model_id_list)
            meta_model_def_dict[meta_model_name] = (mm_id, mm_def)

        return meta_model_def_dict

    def init(self, meta_model_name, model_id_list):
        '''
        @param meta_model_name: Name of the new meta model. Usually the name that is represented.
        @param model_id_list: List of elementary model ids that the meta model
            will be build from
        '''
        #
        # Write SVQ feature representations for each model
        #
        # 1. Filter for all model_id what are defined within the same document
        # 2. Build svq representations for all of them
        #
        logger = logging.getLogger('MetaModelInitializer::__init__')
        logger.info('Estimating meta model %s from elementary models [ %s ]',
                    meta_model_name, ', '.join(model_id_list))
        meta_model_dict = {}
        meta_model_svqmeta_dict = {}
        mod_dict = self.__doclist_model_dict
        document_set = set(mod_dict[m_id][0] for m_id in model_id_list)
        for document_name in sorted(document_set):
            logger.info('Extracting features from document %s', document_name)
            model_svq_gen = self.__load_model_svq_gen(document_name)
            doc_model_dict = {m_id:m_def for m_id, m_def in mod_dict.iteritems()
                              if m_def[0] == document_name and
                                 m_id in model_id_list}
            doc_model_svqmeta_dict = model_svq_gen.write_model_svq(doc_model_dict)
            meta_model_dict.update(doc_model_dict)
            meta_model_svqmeta_dict.update(doc_model_svqmeta_dict)

        #
        # Create meta model specification (meta_model_id, meta_model_def)
        # Generate meta_model_id
        #
        meta_model_doc = self.__context_id
        meta_model_id = ModelDefinitions.get_model_id(meta_model_doc,
                                                      meta_model_name,
                                                      model_index=0)

        #
        # Create meta model definition
        # 1. Estimate average patch size
        # 2. Create model definition tuple
        #
        # m_def[2][1][0]-m_def[2][0][0] computes the models' width
        # m_def[2][1][1]-m_def[2][0][1] the models' height, respectively
        model_size_list = [[m_def[2][1][0] - m_def[2][0][0],
                            m_def[2][1][1] - m_def[2][0][1]]
                           for m_def in meta_model_dict.itervalues()]
        model_size_arr = np.array(model_size_list)
        model_size_avg = np.mean(model_size_arr, axis=0)
        meta_model_size_arr = np.around(model_size_avg)
        meta_model_size_lst = [int(x) for x in meta_model_size_arr]
        meta_model_bounds = ((0, 0), tuple(meta_model_size_lst))
        meta_model_def = (meta_model_doc, meta_model_name, meta_model_bounds)

        #
        # Estimate meta model from svq representations
        #
        meta_model_dict = {meta_model_id : tuple(model_id_list)}
        self.__model_estimator.estimate_model(meta_model_dict,
                                              meta_model_svqmeta_dict,
                                              self.__cleanup_training)

        return meta_model_id, meta_model_def

    def __load_model_svq_gen(self, document_name):

        vw_offset = self.__config.vw_offset
        vw_grid_spacing = self.__config.vw_grid_spacing
        patch_shift_denom = self.__config.patch_shift_denom
        model_patch_size_reduction = self.__config.model_patch_size_reduction
        quantization_path = self.__config.get_quantization_path()
        vw_reader = VisualWordsIO(quantization_path)
        xyvw_array = vw_reader.read_visualwords(document_name,
                                                vw_offset)

        patch_gen = self.__patch_gen_class.load(self.__config, xyvw_array)
        patch_seq = PatchSequenceGenerator(patch_gen.document_bounds())

        vw_grid_bounds = patch_seq.get_vw_grid_bounds()
        vw_grid_ul = vw_grid_bounds[0]
        patch_quantizer = PatchQuantizer(vw_grid_spacing, vw_grid_ul,
                                         patch_shift_denom,
                                         model_patch_size_reduction)

        model_path = self.__config.get_document_model_path(document_name)
        if not exists(model_path):
            makedirs(model_path)
        svq_path = self.__config.get_svq_path(document_name)
        if not exists(svq_path):
            makedirs(svq_path)

        model_svq_gen = ModelSVQGenerator(patch_gen, patch_quantizer,
                                          model_path, svq_path)
        return model_svq_gen

class WhitespaceModels(object):

    def __init__(self, ws_config, ws_model_height_list=(0,),
                 context_id=None, mm_model_id_prefix=''):

        if not ws_config.has_patch_hyp_config():
            raise ValueError('Config does not have patch_hyp_config')

        if len(ws_model_height_list) == 0:
            raise ValueError('model height list must not be empty')

        self.__ws_config = ws_config
        self.__ccs_config = self.__ws_config.patch_hyp_config
        self.__ws_model_height_arr = np.array(ws_model_height_list)[:, None]
        if context_id is None:
            self.__ws_context_id = self.__ws_config.get_ws_context_id()
        else:
            self.__ws_context_id = context_id

        self.__mm_model_id_prefix = mm_model_id_prefix
        self.__cleanup_training = False

    def load_ws_params(self, m_height=0, m_state_index=None):

        m_name_left, m_name_right = ConceptDefinitions.WS_MODEL_IDS
        ws_m_left_tup = self.__load_ws_model_params(mode='left',
                                                    height=m_height,
                                                    m_state_index=m_state_index)
        ws_m_right_tup = self.__load_ws_model_params(mode='right',
                                                    height=m_height,
                                                    m_state_index=m_state_index)
        ws_params = {}
        ws_params[m_name_left] = ws_m_left_tup
        ws_params[m_name_right] = ws_m_right_tup

        return ws_params

    def ws_models_exist(self, m_height=0):
        def check_model_exists(mode):
            ws_model_name = self.__get_ws_model_name(mode, m_height)
            ws_model_path = self.__get_ws_model_def_path(ws_model_name)
            return True if exists(ws_model_path) else False

        return True if (check_model_exists(mode='left') and
                        check_model_exists(mode='right')) else False


    def __load_ws_model_params(self, mode, height, m_state_index):

        ws_model_name = self.__get_ws_model_name(mode, height)
        mm_id, mm_def = self.load_ws_model_def(ws_model_name)
        m_reader = MMWordModelReader(self.__ws_config,
                                     model_state_index=m_state_index)
        ws_concept_def = '%s ;' % mm_id
        ws_concept = m_reader.read_mm_concept(m_document=mm_def[0],
                                              m_id=mm_id,
                                              m_concept_def=ws_concept_def)

        n_transitions = self.__ws_config.get_n_transitions()
        n_states = ws_concept.n_states()
        model_size = self.__ws_config.model_size()
        model_topology = self.__ws_config.model_topology
        transitions_arr = np.zeros((n_states, n_transitions))
        mixtures_arr = np.zeros((n_states, model_size))
        ws_concept.export_state_space(transitions_arr, mixtures_arr)
        return transitions_arr, mixtures_arr, model_topology


    def process_document_list(self, document_list, region_hyps=None):

        logger = logging.getLogger('WhitespaceModels::process_document_list')

        if region_hyps is None:
            if not self.__ws_config.use_patch_hyp_config():
                raise ValueError('Config does not support ccspace_regions')
            # Obtain region statistics
            logger.info('computing CC regions')
            region_hyps = self.ccs_region_hyps(document_list)

            height_quant = self.__ccs_config.region_height_quant
            region_hyps.prune_regions_by_height(height_quant)

        min_height, max_height = region_hyps.minmax_regions_height()
        height_thresh_tup = (min_height - 1, max_height + 1)

        v_cell_size = self.__ws_config.desc_cell_size
        v_cell_cols = self.__ws_config.desc_cell_struct[1]

        # Collect elementary whitespace model definitions
        logger.info('computing whitespace regions')
        logger.info('   cell size = %g', v_cell_size)
        logger.info('   cell columns = %g', v_cell_cols)
        logger.info('   height in ]%g, %g[', *height_thresh_tup)
        doc_model_dict = {}

        for document_id in document_list:
            vs_left, vs_right, vs_bounds = self.whitespace_voting(document_id,
                                                                  region_hyps,
                                                                  v_cell_size,
                                                                  v_cell_cols)
            vs_ul_xy = vs_bounds[:2]
            model_dict_tup = self.whitespace_models(document_id,
                                                    vs_left, vs_right,
                                                    vs_ul_xy,
                                                    height_thresh_tup)
            doc_model_left_dict, doc_model_right_dict = model_dict_tup
            doc_model_dict.update(doc_model_left_dict)
            doc_model_dict.update(doc_model_right_dict)

        # Group elementary models into meta models
        meta_model_id_set = set(m_def[1] for m_def in doc_model_dict.values())
        meta_model_id_list = sorted(meta_model_id_set)
        logger.info('grouping whitespace regions according to height: %s',
                    ' '.join(meta_model_id_list))
        meta_model_dict = {}
        for meta_model_id in meta_model_id_list:
            meta_m_id_list = [m_id for m_id, m_def in doc_model_dict.items()
                              if m_def[1] == meta_model_id]
            meta_model_dict[meta_model_id] = meta_m_id_list

        # Estimate meta models
        logger.info('computing whitespace models')
        meta_model_init = MetaModelInitializer(self.__ws_config,
                                               document_list=None,
                                               doc_model_dict=doc_model_dict,
                                               context_id=self.__ws_context_id,
                                               cleanup_training=self.__cleanup_training)
        mm_def_dict = meta_model_init.init_models(meta_model_dict)

        for mm_id, mm_def in sorted(mm_def_dict.values()):
            self.store_ws_model_def(mm_id, mm_def)

    def __get_ws_model_def_path(self, model_name):
        model_doc_path = self.__ws_config.get_document_model_path(self.__ws_context_id)
        whitespace_model_path = ''.join([model_doc_path,
                                         '%s_def.pickle' % model_name])
        return whitespace_model_path

    def store_ws_model_def(self, mm_id, mm_def):
        logger = logging.getLogger('WhitespaceModels::store_ws_model_def')
        whitespace_model_path = self.__get_ws_model_def_path(mm_def[1])
        logger.info('storing meta model definitions: %s', whitespace_model_path)
        with open(whitespace_model_path, 'wb') as fd:
            pickle.dump((mm_id, mm_def), fd)

    def load_ws_model_def(self, model_name):
        logger = logging.getLogger('WhitespaceModels::load_ws_model_def')
        whitespace_model_path = self.__get_ws_model_def_path(model_name)
        logger.info('loading meta model definitions: %s', whitespace_model_path)
        with open(whitespace_model_path, 'rb') as fd:
            mm_id, mm_def = pickle.load(fd)
        return mm_id, mm_def

    def ccs_region_hyps(self, document_list):
        logger = logging.getLogger('WhitespaceModels::ccs_region_hyps')
        logger.info('--- WSBoF - ConnectedComponentSpace ---')
        region_hyps = RegionHypotheses()
        for document_id in document_list:
            logger.info('[ Processing %s ]', document_id)
            cc_space = WSBoFCCSpace(self.__ws_config)
            region_hyps.extend_region_list(document_id,
                                            cc_space.region_list(document_id))


        return region_hyps

    @staticmethod
    def whitespace_voting(document_id, region_hyps, v_cell_size, v_cell_cols):

        logger = logging.getLogger('WhitespaceModels::whitespace_voting')
        logger.info('[ Processing %s ]', document_id)

        v_cell_cols_half = int(max(v_cell_cols / 2, 1))

        document_region_list = region_hyps.filter_region_list(document_id)
        bounds_arr = RegionProcessor.bounds_arr(document_region_list)
        xy_min_arr = np.amin(bounds_arr[:, :2], axis=0)
        xy_max_arr = np.amax(bounds_arr[:, 2:], axis=0)
        bounds_area = xy_max_arr - xy_min_arr
        voting_space_bounds = xy_min_arr.tolist() + xy_max_arr.tolist()
        voting_space_left = np.zeros(bounds_area[[1, 0]])
        voting_space_right = np.zeros(bounds_area[[1, 0]])
        for bbox in bounds_arr:
            ul_row = bbox[1] - xy_min_arr[1]
            ul_col = bbox[0] - xy_min_arr[0]
            lr_row = bbox[3] - xy_min_arr[1]
            lr_col = bbox[2] - xy_min_arr[0]

            # Downvoting for inner area
            voting_space_left[ul_row:lr_row, ul_col: lr_col] -= 2
            voting_space_right[ul_row:lr_row, ul_col: lr_col] -= 2


            # Upvote outer areas (left side, right side)
            # The inner area at the region bound stays neutral
            #
            # Voting area depends on the descriptor size
            # (cell size and cell colls)
            # Therefore, descriptors will tend to be covering only
            # whitespace areas

            # Left side
            ul_col_outer_f = ul_col - v_cell_cols_half * v_cell_size
            ul_col_outer_c = ul_col - v_cell_size
            if ul_col_outer_f > 0 and ul_col_outer_c > 0:
                voting_space_left[ul_row:lr_row,
                                  ul_col_outer_f:ul_col_outer_c + 1] += 1

            # Right side
            lr_col_outer_c = lr_col + v_cell_size
            lr_col_outer_f = lr_col + v_cell_cols_half * v_cell_size
            if (lr_col_outer_c < voting_space_right.shape[1] and
                lr_col_outer_c < voting_space_right.shape[1]):
                voting_space_right[ul_row:lr_row,
                             lr_col_outer_c:lr_col_outer_f + 1] += 1

        # Threshold regions
        voting_space_left[voting_space_left <= 0] = 0
        voting_space_left[voting_space_left > 0] = 1
        voting_space_right[voting_space_right <= 0] = 0
        voting_space_right[voting_space_right > 0] = 1

        return (voting_space_left,
                voting_space_right,
                voting_space_bounds)

    def whitespace_models(self, document_id,
                          voting_space_left,
                          voting_space_right,
                          voting_space_ul_xy,
                          height_thresh_tup):
        ws_config = self.__ws_config
        vw_grid_spacing_x = ws_config.vw_grid_spacing[0]
        logger = logging.getLogger('WhitespaceModels::whitespace_models')
        logger.info('estimating regions from voting space')

        vw_reader = VisualWordsIO(ws_config.get_quantization_path())
        xyvw_array = vw_reader.read_visualwords(document_id,
                                                ws_config.vw_offset)
        document_bounds = VisualWordMatGenerator.document_bounds(xyvw_array)
        vw_grid_ul = tuple(document_bounds[:2])
        patch_quant = PatchQuantizer(ws_config.vw_grid_spacing,
                                     vw_grid_ul,
                                     ws_config.patch_shift_denom,
                                     patch_size_reduction=(0, 0, 0, 0),
                                     patch_width_expansion=0.0)
        logger.info('left-side white space')
        mode = 'left'
        # left, right
        # left side component: snap region at the right side towards the comp.
        desc_offset_tup = (vw_grid_spacing_x, 0)
        doc_model_left_dict = self.__document_model_dict(patch_quant,
                                                         document_id,
                                                         height_thresh_tup,
                                                         document_bounds,
                                                         voting_space_left,
                                                         voting_space_ul_xy,
                                                         mode, desc_offset_tup)
        logger.info('right-side white space')
        mode = 'right'
        # left, right
        # right side component: snap region at the left side towards the comp.
        desc_offset_tup = (0, vw_grid_spacing_x)
        doc_model_right_dict = self.__document_model_dict(patch_quant,
                                                          document_id,
                                                          height_thresh_tup,
                                                          document_bounds,
                                                          voting_space_right,
                                                          voting_space_ul_xy,
                                                          mode, desc_offset_tup)
        return doc_model_left_dict, doc_model_right_dict

    def __document_model_dict(self, patch_quant, document_id,
                              height_thresh_tup, document_bounds,
                              voting_space, voting_space_ul_xy,
                              mode, desc_offset_tup):
        im_label, _ = mh.label(voting_space, Bc=np.ones((3, 3)))
        label_index = im_label_index((0, im_label))

        document_model_dict = {}
        for model_index, cc_prop in label_index.items():
            cc_bounds = cc_prop.bounds()
            # Normalize voting space coordinates to image coordinates
            cc_bounds = ((cc_bounds[0][0] + voting_space_ul_xy[0],
                          cc_bounds[0][1] + voting_space_ul_xy[1]),
                         (cc_bounds[1][0] + voting_space_ul_xy[0],
                          cc_bounds[1][1] + voting_space_ul_xy[1]))
            cc_height = cc_bounds[1][1] - cc_bounds[0][1]
            if (cc_height > height_thresh_tup[0] and
                cc_height < height_thresh_tup[1] and
                cc_bounds[0][0] >= document_bounds[0] and
                cc_bounds[0][1] >= document_bounds[1] and
                cc_bounds[1][0] <= document_bounds[2] and
                cc_bounds[1][1] <= document_bounds[3]):

                cc_bnds_snp = patch_quant.snap_bounds_to_grid(cc_bounds)
                m_id, m_def = self.__get_model_def(document_id,
                                                   model_index,
                                                   cc_bnds_snp,
                                                   mode,
                                                   desc_offset_tup)
                document_model_dict[m_id] = m_def

        return document_model_dict


    def __get_model_def(self, document_id, model_index,
                        cc_bnds_snp, mode, desc_offset_tup):
        ws_bnds = ((cc_bnds_snp[0][0] - desc_offset_tup[1], cc_bnds_snp[0][1]),
                   (cc_bnds_snp[1][0] + desc_offset_tup[0], cc_bnds_snp[1][1]))
        ws_size_snp = (cc_bnds_snp[1][0] - cc_bnds_snp[0][0],
                       cc_bnds_snp[1][1] - cc_bnds_snp[0][1])
#         logger.info('ws_size: %s', str(ws_size_snp))
        model_name = self.__get_ws_model_name(mode, height=ws_size_snp[1])
        m_id = ModelDefinitions.get_model_id(document_id,
                                             model_name,
                                             model_index)
        m_def = (document_id, model_name, ws_bnds)

        return m_id, m_def

    def __get_ws_model_name(self, mode, height):
        ws_height_quant = self.__quantize_height(height)
        mm_model_id_sep = '_' if self.__mm_model_id_prefix != '' else ''
        mm_model_id_prefix = self.__mm_model_id_prefix + mm_model_id_sep
        model_name = '%sws_%s_h%d' % (mm_model_id_prefix,
                                      mode,
                                      ws_height_quant)
        return model_name

    def __quantize_height(self, height_snp):
        height_arr = np.array([[height_snp]])
        dist_arr = scipy.spatial.distance.cdist(height_arr,
                                                self.__ws_model_height_arr)
        model_heigt_idx = np.argmin(dist_arr, axis=1)
        model_height = self.__ws_model_height_arr[model_heigt_idx, 0]
        return model_height

