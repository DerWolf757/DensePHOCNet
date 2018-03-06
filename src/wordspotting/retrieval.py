'''
Created on Jan 27, 2013

@author: leonard
'''
import logging
import time
import numpy as np
# from patrec.spatial.distance import cdist
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import minimum_filter, gaussian_filter
from patrec.serialization.matrix_io import MatrixIO
from patrec.evaluation.rectangle_intersection import RectangleIntersection
from esmeralda.align import MMAlign, MMHypChain

class RetrievalResults(object):

    def __init__(self, config, document_name):
        self.__config = config
        self.__document_name = document_name
        self.__retrieval_mat = None
        self.__score_mat = None

    def register_retrieval_mat(self, query_model_id, retrieval_mat):
        self.__retrieval_mat = retrieval_mat
        if self.__config.store_retrieval_mat:
            retrieval_mat_filepath = self.__config.get_retrieval_mat_filepath(self.__document_name,
                                                                              query_model_id)
            MatrixIO.store_matrix_as_byte(retrieval_mat, retrieval_mat_filepath)

    def register_score_mat(self, query_model_id, score_mat, height_mat):
        self.__score_mat = score_mat
        if self.__config.store_score_mat:
            score_mat_filepath = self.__config.get_score_mat_filepath(self.__document_name,
                                                                      query_model_id)
            np.savez(score_mat_filepath,
                     score_mat=score_mat,
                     height_mat=height_mat)

    def retrieval_mat(self):
        if self.__retrieval_mat is None:
            raise ValueError('No retrieval_mat has been registered, yet!')
        return self.__retrieval_mat

    def score_mat(self):
        if self.__score_mat is None:
            raise ValueError('No score_mat has been registered, yet!')
        return self.__score_mat


class PatchScoreFilterConfig(object):

    def __init__(self, smooth_scale, nms_scale, ifsaccu_filter=None):
        """Set config parameters

        Params:
            smooth_scale: Scale factor * query_size -1 is the spatial extend
                of the Gaussian for score smoothing
            nms_scale: Scale factor * query_size -1 is the spatial extend of
                the non-min-suppression filter for generating hypotheses.
                nms_scale=2 does not allow for (largely) overlapping hypotheses.
            ifsaccu_filter: Size of accu dilation filter that has been applied
                after first stage inverted file structure decoding for expanding
                the search space. Given as tuple (rows, cols).
                Optional, defaults to None.
                The filter size is important to adjust Gaussian score smoothing
                during patch score pruning.
        """
        self.smooth_scale = smooth_scale
        self.nms_scale = nms_scale
        self.ifsaccu_filter = ifsaccu_filter

    def has_ifsaccu_filter(self):
        return (self.ifsaccu_filter is not None and
                self.ifsaccu_filter[0] >= 1 and
                self.ifsaccu_filter[1] >= 1)

class RetrievalGenerator(object):


    def __init__(self, retrieval_results, patch_quantizer, gt_doc_model_dict,
                 patches_thresh, patch_overlap, patch_nms,
                 patch_hyp_fitter=None):
        """Constructor for setting up class attributes

        Params:
            retrieval_results: Object implementing RetrievalResults (see above)
                interface
            patch_quantizer: Object implementing
                bofhwr.features.patch_termvector_sequence.PatchQuantizer interface
            gt_doc_model_dict: Dictionary of the words contained in the
                document processed in this Wordspotting instance
                --> Ground Truth
            patches_thresh: Threshold for the number of patches to return
                per query and page
            patch_overlap: Factor ([0,1]) for determining when a patch is
                relevant with respect to the ground truth
            patch_nms: Non-minimum-suppression parameter.
                A 2-tuple specifying the filter size.
                OR
                (0,0) for computing the filter size dynamically from patch size
                and patch shift
            patch_hyp_fitter: Object implementing
                process_patch( patch_bounds )
                with patch_bounds ((ul_x, ul_y), lr_x, lr_y))
                OR None
                process_patch should fit the patch hypothesis to the text in the
                document
        """
        self.__retrieval_results = retrieval_results
        self.__patch_quantizer = patch_quantizer
        self.__patches_thresh = patches_thresh
        self.__patch_overlap = float(patch_overlap)
        self.__patch_nms = patch_nms
        self.__gt_doc_model_dict = gt_doc_model_dict
        if patch_hyp_fitter is None:
            self.__patch_fitter = lambda x:x
        else:
            self.__patch_fitter = patch_hyp_fitter.process_patch


    def generate_retrieval_matrix(self, query_model_id, query_model_def,
                                  patch_score_mat, patch_align_mat,
                                  patch_height_mat, patch_yx_mat,
                                  score_filter_config):
        """Generates and stores the retrieval matrix for each query

        A 6 x max(N,patches_thresh), (N is the number of extracted patches)
        matrix containing (row-wise):
        score, bin-relevance, ul_x-coord, ul_y-coord, lr_x-coord, lr_y-coord
        is registered in self.__retrieval_results

        Entries in the retrieval result matrix are sorted
        according to their score values.

        Params:
            query_model_id: Identifier (in model_dict) of the model
                that will be processed
            query_model_def: Definition (from model_dict) of the model
                that will be processed
            patch_score_mat: ndarray containing the hmm scores
                in the current patch layout (for the current query and document)
            patch_align_mat: 3D ndarray containing patch alignments
                (start_offset, end_offset) for every patch.
                Start offset refers to adding the offset to the start index of
                the patch. Stop offset refers to subtracting the offset from the
                end index of the patch. --> can be a 3D ndarray of zeros
            patch_height_mat: ndarray containing the height for all patches
            patch_yx_mat: ndarray containing all patch y,x coordinates.
            score_filter_config: Object of type PatchScoreFilterConfig
                specifying parameters for score smoothing and non-min-
                suppression
        """
        rect_intersect = RectangleIntersection()

        #
        # Filter ground truth for query_name
        #
        query_name = query_model_def[1]
        gt_doc_model_query_bounds = [x[2]
                                     for x in self.__gt_doc_model_dict.values()
                                     if x[1] == query_name]


        #
        # Load patch_score_mat
        #
        query_bounds = query_model_def[2]

        p_quant_tup = self.__patch_quantizer.get_patch_size_snp(query_bounds,
                                                                return_displacement=True)
        # x and y displacements compensate for patch_size_reduction
        query_size_snp, x_displ, y_displ = p_quant_tup
        # Register expanded score_mat
        self.__retrieval_results.register_score_mat(query_model_id,
                                                    patch_score_mat,
                                                    patch_height_mat)

        prune_time_start = time.time()
        # Prune patch_score_mat
        patch_score_mat, patch_thresh_score = self.prune_patch_score_mat(patch_score_mat,
                                                                         query_size_snp,
                                                                         score_filter_config)
        # Shape patch_score_mat vector
        patch_score_vector = patch_score_mat.reshape(patch_score_mat.size)
        pam_shape = patch_align_mat.shape
        pam_lin_shape = (pam_shape[0] * pam_shape[1], pam_shape[2])
        patch_align_lin = patch_align_mat.reshape(pam_lin_shape)
        patch_height_lin = patch_height_mat.reshape(patch_height_mat.size)

        # Only consider scores < patch_thresh_score
        retrieved_patches_mask = patch_score_vector < patch_thresh_score
        retrieved_patches_score = patch_score_vector[retrieved_patches_mask]
        retrieved_patches_align = patch_align_lin[retrieved_patches_mask, :]
        retrieved_patches_height = patch_height_lin[retrieved_patches_mask]
        prune_time_end = time.time()
        prune_time = prune_time_end - prune_time_start

        # Determine number of retrieved patches
        retrieved_patches_num = retrieved_patches_score.size
        # Allocate scores-relevance-x-y data structure for retrieved patches
        retrieved_patches_relevance = np.empty(retrieved_patches_num)
        retrieved_patches_ul_x = np.empty(retrieved_patches_num)
        retrieved_patches_ul_y = np.empty(retrieved_patches_num)
        retrieved_patches_lr_x = np.empty(retrieved_patches_num)
        retrieved_patches_lr_y = np.empty(retrieved_patches_num)

        patch_yx_mat_retrieved = patch_yx_mat[retrieved_patches_mask, :]

        fitting_time = 0

        # Iterate over all patch indices on the page
        for ret_tup in zip(range(retrieved_patches_num),
                           patch_yx_mat_retrieved,
                           retrieved_patches_align,
                           retrieved_patches_height):
            index, (y, x), align_ind_off, height = ret_tup
            start_offset, end_offset = self.__patch_quantizer.get_align_offsets(align_ind_off)
            patch_xy_ul = (x - x_displ[0] + start_offset, y - y_displ[0])
            patch_xy_lr = (x + query_size_snp[0] + x_displ[1] - end_offset,
                           y + height + y_displ[1])
            # Fit patch to hypothesis
            fitting_time_start = time.time()
            patch_xy_ul, patch_xy_lr = self.__patch_fitter((patch_xy_ul,
                                                            patch_xy_lr))
            fitting_time_end = time.time()
            fitting_time += fitting_time_end - fitting_time_start
            # Check if patch is relevant
            patch_rel = rect_intersect.intersects((patch_xy_ul, patch_xy_lr),
                                                  gt_doc_model_query_bounds,
                                                  self.__patch_overlap)
            # Add it to the retrieval list
            retrieved_patches_relevance[index] = patch_rel
            retrieved_patches_ul_x[index] = patch_xy_ul[0]
            retrieved_patches_ul_y[index] = patch_xy_ul[1]
            retrieved_patches_lr_x[index] = patch_xy_lr[0]
            retrieved_patches_lr_y[index] = patch_xy_lr[1]

        rank_time_start = time.time()
        # Sort patches (score / relevance) with respect to their scores
        retrieved_patches_scores_sort_ind = np.argsort(retrieved_patches_score)
        # Cut off index list after patch_thresh patches
        retrieved_patches_scores_sort_ind = retrieved_patches_scores_sort_ind[0:self.__patches_thresh]
        retrieved_patches_scores_sort = retrieved_patches_score[retrieved_patches_scores_sort_ind]
        retrieved_patches_relevance_sort = retrieved_patches_relevance[retrieved_patches_scores_sort_ind]
        retrieved_patches_ul_x_sort = retrieved_patches_ul_x[retrieved_patches_scores_sort_ind]
        retrieved_patches_ul_y_sort = retrieved_patches_ul_y[retrieved_patches_scores_sort_ind]
        retrieved_patches_lr_x_sort = retrieved_patches_lr_x[retrieved_patches_scores_sort_ind]
        retrieved_patches_lr_y_sort = retrieved_patches_lr_y[retrieved_patches_scores_sort_ind]
        rank_time_end = time.time()
        rank_time = rank_time_end - rank_time_start
        retrieved_patches = np.vstack((retrieved_patches_scores_sort,
                                       retrieved_patches_relevance_sort,
                                       retrieved_patches_ul_x_sort,
                                       retrieved_patches_ul_y_sort,
                                       retrieved_patches_lr_x_sort,
                                       retrieved_patches_lr_y_sort))
        self.__retrieval_results.register_retrieval_mat(query_model_id,
                                                        retrieved_patches)
        return prune_time + fitting_time + rank_time


    def prune_patch_score_mat(self, patch_score_mat, query_size_snp,
                              score_filter_config):
        '''
        Prune by non-minimum suppression
        '''

        # upper score bound (worst) has been used for initializing score mat
        # --> use max
        patch_thresh_score = patch_score_mat.max()

        if self.__patch_nms != (0, 0):
            raise ValueError('patch_nms != (0, 0) is unsupported')
        #
        # Generate filter for smoothing and non-minimum suppression
        #
        query_shift_snp = self.__patch_quantizer.get_patch_shift_snp(query_size_snp)

        #
        # Smooth by applying a Gaussian filter
        # Attention: large smoothing filter will only work for ifs decoding
        # vt-scores are log-probs which leads to a weighted joint probability
        # after Gaussian smoothing. Incorporating too many low-probs will make
        # the joint probability indiscriminative
        smooth_filter_scale = score_filter_config.smooth_scale
        smooth_filter = (smooth_filter_scale * np.array(query_size_snp) /
                        np.array(query_shift_snp)) - 1
        smooth_filter = np.array(smooth_filter[[1, 0]], dtype=np.int)
        if score_filter_config.has_ifsaccu_filter():
            ifsaccu_filter = score_filter_config.ifsaccu_filter
            smooth_filter = np.array([max(min(ifs_f, sm_f), 1)
                                      for ifs_f, sm_f in zip(ifsaccu_filter,
                                                             smooth_filter)])
        # Compute the variance for each axis from the desired spatial extend of
        # the filter.
        # Filter size: int(smooth_filter/9.0 * 4 + 0.5) * 2 + 1
        # 4/9 < 0.5 ==> filter size will be floored in order to be uneven
        gaussian_filter_sigma = smooth_filter.astype(float) / 9.0
        patch_score_mat[:, :] = gaussian_filter(patch_score_mat,
                                                gaussian_filter_sigma,
                                                truncate=4.0)

        #
        # Prune by non-minimum-suppression
        #
        nms_filter_scale = score_filter_config.nms_scale
        nms_filter = (nms_filter_scale * np.array(query_size_snp) /
                      np.array(query_shift_snp)) - 1
        nms_filter = np.array(nms_filter[[1, 0]], dtype=np.int)

        patch_min_filter = minimum_filter(patch_score_mat, nms_filter)
        patch_score_localmin = patch_min_filter == patch_score_mat

        #
        # patch_score_mask contains all elements
        #     NOT (being locally optimal)
        #
        patch_score_mask = np.logical_not(patch_score_localmin)
        patch_score_mat[patch_score_mask] = patch_thresh_score

        return patch_score_mat, patch_thresh_score



class PatchRetrievalProcessor(object):


    def __init__(self, retrieval_generator, model_list, doc_model_dict,
                 model_concept_dict, use_bg_norm, model_concept_name_dict,
                 model_patchmask_dict,
                 patch_yx_mat, patch_mat_dim, patch_quantizer,
                 patch_score_bound, vt_beamwidth, vt_problow):
        """
        Initialize data structures needed for an integrated generation, score
        computation and retrieval of patch representations.

        Params:
            retrieval_generator: Object implementing generate_retrieval_matrix
                Must be capable of saving matrix to disk
            model_list: List of model identifiers (m_id) that are going to be decoded
            doc_model_dict: Dictionary saving model definitions for decoding
            model_concept_dict: Dictionary of MMConcept objects (values) for
                model ids (keys)
            use_bg_norm: Boolean flag indicating whether to perform score
                normalization by subtracting bg-ws from query path log-prob.
            model_concept_name_dict: Dictionary saving mm concept names
                --> the decoding target for each model
            model_patchmask_dict: Dictionary with boolean patch mask arrays
                for masking patches for HMM decoding
            patch_yx_mat: nd_array containing patch coordinates
            patch_mat_dim: Tuple (rows, cols) of patch matrix
            patch_quantizer: PatchQuantizer object for snapping query patch size
                in order to align with dense grid
            patch_score_bound: Lower bound for patch score, given as neg-log
                value --> the worse the patch score, the larger the value
            vt_beamwidth: Viterbi decoding beam width
            vt_problow: Viterbi decoding floor probability
        """
        logger = logging.getLogger('PatchRetrievalProcessor::__init__')
        self.__retrieval_generator = retrieval_generator
        self.__doc_model_dict = doc_model_dict
        self.__model_list = model_list
        self.__patch_yx_mat = patch_yx_mat
        self.__patch_score_bound = patch_score_bound

        self.__use_bg_norm = use_bg_norm
        logger.info('bg/ws score normalization: %s', str(self.__use_bg_norm))

        self.__model_concept_name_list = []
        self.__decoder_list = []
        self.__score_mat_list = []
        self.__height_mat_list = []
        self.__align_mat_list = []
        # See below for patchmask_mat class member definition
        model_patchmask_mat_list = []

        logger.info('Going to initialize MMAlign decode objects:\n%s',
                    ', '.join(model_list))

        for m_id in model_list:
            #
            # Initialize decoders
            #
            mm_concept = model_concept_dict[m_id]
            decoder = MMAlign(mm_concept, vt_beamwidth, vt_problow)
            self.__decoder_list.append(decoder)

            #
            # Model bounds and size (snapped to visual word grid)
            #

            model_def = doc_model_dict[m_id]
            m_bounds = model_def[2]
            m_size_snp = patch_quantizer.get_patch_size_snp(m_bounds)
            m_concept_name = model_concept_name_dict[m_id]
            self.__model_concept_name_list.append(m_concept_name)
            #
            # Patch mask definitions
            #
            m_patchmask_mat = model_patchmask_dict[m_id]
            model_patchmask_mat_list.append(m_patchmask_mat)

            #
            # Initialize score vectors
            #
            # initialize score mat for current model
            score_mat = np.ones(patch_mat_dim, dtype=np.float64)
            # set initial value to worst score (lower bound)
            score_mat *= patch_score_bound
            self.__score_mat_list.append(score_mat)

            #
            # Initialize height masks
            #
            # The default value is the models actual patch height
            height_mat = (np.ones(patch_mat_dim, dtype=np.int16) *
                          m_size_snp[1])
            self.__height_mat_list.append(height_mat)

            #
            # Patch alignment indices
            # [off_start, off_end] given as number of termvectors (-> offset at
            # beginning and end)
            align_mat_dim = list(patch_mat_dim)
            align_mat_dim.append(2)
            align_mat = np.zeros(align_mat_dim, np.int)
            self.__align_mat_list.append(align_mat)

        # Stack patch masks along the third dimension. This way a mask vector
        # for all models at a given patch position can be extracted easily [row,col,:]
        self.__model_patchmask_mat = np.dstack(tuple(model_patchmask_mat_list))

        # Counter for keeping track of the number of patches that have been
        # decoded with the Viterbi algorithm
        self.__n_patches_decoded = 0


    def patchmask_mat(self):
        '''Reduce 3rd dimension
        __model_patch_mat mask contains a layer/channel (3rd dim) per model that
        is being processed in the same patch size retrieval run.
        By setting an element to TRUE if any of the elements along axis 2 (3rd dim)
        is TRUE a patch will be decoded if it is relevant to any model.
        '''
        patchmask_mat = np.any(self.__model_patchmask_mat == 1, axis=2)
        return patchmask_mat

    def process_termvectors(self, svq_termvectors,
                            _,
                            patch_row, patch_col,
                            patch_height):
        '''
        '''
        use_bg_norm = self.__use_bg_norm
        patch_mask_vec = self.__model_patchmask_mat[patch_row, patch_col, :]

        for m_tup in zip(self.__model_concept_name_list,
                         self.__decoder_list,
                         self.__score_mat_list,
                         self.__align_mat_list,
                         self.__height_mat_list,
                         patch_mask_vec):
            (m_concept_name,
             m_decoder,
             m_score_mat,
             m_align_mat,
             m_height_mat,
             m_patch_mask) = m_tup

            if m_patch_mask == 1:
                # ATTENTION: returns -1 if no score could be obtained
                # (also see initialization)
                hypchain = MMHypChain()
                score_new = m_decoder.viterbi_decoding(svq_termvectors,
                                                       hypchain)
                # Increase counter for decoded patches
                self.__n_patches_decoded += 1

                # Skip iteration if no Viterbi path could be found
                # (depends on Viterbi pruning parameters)
                if score_new is None:
                    continue

                # Obtain decoding result
                #
                # List of model names
                hypmodel_list = hypchain.modelname_list()
                # List of frame alignment tuples [(m0_start, mo_end), ...]
                # Indices are BOTH inclusive
                hypalign_list = hypchain.modelalign_list()
                # List of cumulative decoding path scores (negative logarithmic)
                hypcscore_list = hypchain.modelcscore_list()
                # Only store alignment information,
                # if background / whitespace model decoding available
                # Esmeralda hypotheses alignments are always inclusive
                # offsets refer to number of frames being removed at the
                # beginning and end of the patch
                m_hyp_idx = hypmodel_list.index(m_concept_name)
                # first frame of model segment
                start_offset = hypalign_list[m_hyp_idx][0]
                # last of last segment - last frame of model segment
                # --> number of frames in sequence after model segment
                end_offset = hypalign_list[-1][1] - hypalign_list[m_hyp_idx][1]
                m_align_mat[patch_row, patch_col, :] = [start_offset,
                                                        end_offset]
                # If score normalization is activated, it is performed according
                # to the bg-ws model scores for the partial sequence,
                # the query model score and the number of frames that the query
                # model has been aligned with
                #
                if use_bg_norm:
                    # Number of frames that the query models has been aligned
                    # with, including whitespace models
                    # --> hyp_idx_start - 1, hyp_idx_end + 1
                    hyp_idx_start = m_hyp_idx# - 1
                    hyp_idx_end = m_hyp_idx# + 1
                    # hypalign indices are both inclusive (start and end)
                    # --> end_index +1 - start_index
                    n_m_frames = ((hypalign_list[hyp_idx_end][1] + 1) -
                                  hypalign_list[hyp_idx_start][0])
                    # Pad cumulative score list with an initial zero.
                    # --> Indices will be shifted by one ( + 1 )
                    hypcscore_list.insert(0, 0)
                    # Obtain score for query model (without bg model scores)
                    score_new = (hypcscore_list[hyp_idx_end + 1] -
                                 hypcscore_list[hyp_idx_start])
                    # Normalize with respect to the number of frames that have
                    # been aligned with the query.
                    score_new /= n_m_frames
                else:
                    # Total number of frames that have been aligned within
                    # the patch.
                    n_t_frames = ((hypalign_list[-1][1] + 1) -
                                  hypalign_list[0][0])
                    score_new /= n_t_frames

                score_current = m_score_mat[patch_row, patch_col]
                # Skip the iteration if the new score is worse than the current
                # score.
                # Note: scores are initialized with a lower (higher vales are
                # worse) patch score estimate
                if score_new >= score_current:
                    continue
                m_score_mat[patch_row, patch_col] = score_new
                # Store patch height information
                m_height_mat[patch_row, patch_col] = patch_height

    def n_patches_decoded(self):
        return self.__n_patches_decoded

    def retrieve_patches(self, ifs_accu_filter=None):
        '''
        '''
        score_filter_config = PatchScoreFilterConfig(smooth_scale=1.0,
                                                     nms_scale=2.0,
                                                     ifsaccu_filter=ifs_accu_filter)
        ret_time = 0
        for m_id, m_score_mat, m_align_mat, m_height_mat in zip(self.__model_list,
                                                              self.__score_mat_list,
                                                              self.__align_mat_list,
                                                              self.__height_mat_list):
            m_def = self.__doc_model_dict[m_id]


            ret_time += self.__retrieval_generator.generate_retrieval_matrix(m_id,
                                                                             m_def,
                                                                             m_score_mat,
                                                                             m_align_mat,
                                                                             m_height_mat,
                                                                             self.__patch_yx_mat,
                                                                             score_filter_config)
        return ret_time

class IFSAccuRetrievalProcessor(object):


    def __init__(self, retrieval_generator, model_list, doc_model_dict,
                 model_vw_list, patch_quant, ifs_accu):
        """Initialize data structures needed for an integrated generation, score
        computation and retrieval of patch representations.

        Params:
            retrieval_generator: Object implementing generate_retrieval_matrix
                Must be capable of saving matrix to disk
            doc_model_dict: dictionary saving model definitions for decoding
            model_list: List of model identifiers (m_id) that are going to
                be decoded
            ifs_accu: Representing the documents inverted file
                structure accumulator
        """

        self.__retrieval_generator = retrieval_generator
        self.__model_list = model_list
        self.__doc_model_dict = doc_model_dict
        self.__ifs_accu = ifs_accu
        self.__model_vw_list = model_vw_list
        self.__patch_quant = patch_quant

        self.__score_mat_list = []

        align_mat_dim = list(ifs_accu.accu_mat_dim())
        align_mat_dim.append(2)
        self.__align_mat_dummy = np.zeros(align_mat_dim, np.int)
        self.__score_filter_config = PatchScoreFilterConfig(smooth_scale=2.0,
                                                            nms_scale=1.0)

    def process_ifs_model_accus(self):
        '''
        '''
        for model_vw_arr in self.__model_vw_list:
            accu_mat = self.__ifs_accu.accumulate_ght(model_vw_arr)
#             accu_mat = self.__ifs_accu.accumulate_ght_py(model_vw_arr)
            # Invert accumulator scores:
            # --> large negative values are good, small negative values are bad
            # the worst score is zero: The accumulator cell did not receive any
            # hits
            accu_mat *= -1
            self.__score_mat_list.append(accu_mat)

    def prune_scores(self, query_size_snp):
        score_filter_config = self.__score_filter_config
        for score_mat in self.__score_mat_list:
            res_tup = self.__retrieval_generator.prune_patch_score_mat(score_mat,
                                                                       query_size_snp,
                                                                       score_filter_config)
            score_mat, score_thresh = res_tup
            score_mat_mask = score_mat < score_thresh
            score_mat[:, :] = score_mat_mask


    def filter_scores(self, filter_func, filter_def):
        for score_mat in self.__score_mat_list:
            score_mat[:, :] = filter_func(score_mat, filter_def)

    def model_scoremat_dict(self):
        m_scoremat_dict = {}
        for m_id, m_score_mat in zip(self.__model_list, self.__score_mat_list):
            m_scoremat_dict[m_id] = m_score_mat
        return m_scoremat_dict

    def retrieve_patches(self):
        '''
        '''
        score_filter_config = self.__score_filter_config
        align_mat = self.__align_mat_dummy
        ret_time = 0
        for m_id, m_score_mat in zip(self.__model_list, self.__score_mat_list):
            m_def = self.__doc_model_dict[m_id]
            m_bounds = m_def[2]
            m_bounds_snp = self.__patch_quant.get_patch_size_snp(m_bounds)
            m_height_snp = m_bounds_snp[1]
            height_mat = np.ones_like(align_mat) * m_height_snp
            accu_yx_mat = self.__ifs_accu.accu_yx_mat()
            ret_time += self.__retrieval_generator.generate_retrieval_matrix(m_id,
                                                                             m_def,
                                                                             m_score_mat,
                                                                             align_mat,
                                                                             height_mat,
                                                                             accu_yx_mat,
                                                                             score_filter_config)
        return ret_time

class PatchDistanceRetrievalProcessor(object):

    def __init__(self, retrieval_generator, model_list, doc_model_dict,
                 model_vwmat_list, patch_feat_gen, patch_yx_mat, patch_mat_dim,
                 patch_quant, distance_metric):
        '''
        '''
        logger = logging.getLogger('PatchDistanceRetrievalProcessor::__init__')
        self.__retrieval_generator = retrieval_generator
        self.__doc_model_dict = doc_model_dict
        self.__model_list = model_list
        self.__patch_yx_mat = patch_yx_mat
        self.__patch_mat_dim = patch_mat_dim
        self.__patch_quant = patch_quant
        self.__distance_metric = distance_metric

        if len(model_list) != len(model_vwmat_list):
            raise ValueError('Model id and vwmat list length does not match!')

        n_models = len(model_vwmat_list)
        self.__n_patches = np.prod(np.array(patch_mat_dim))
        sp_dim = patch_feat_gen.model_size()
        self.__m_sp_mat = np.empty((n_models, sp_dim), dtype='float64')

        score_mat_dim = list(patch_mat_dim)
        score_mat_dim.append(n_models)
        # Initialize scores with -1 --> worst score indicator, handled separately
        self.__score_mat = np.ones(score_mat_dim, dtype='float32') * -1

        align_mat_dim = list(patch_mat_dim)
        align_mat_dim.append(2)
        self.__align_mat_dummy = np.zeros(align_mat_dim, np.int)

        self.__buffer_size = 1000
        self.__buffer = np.zeros((self.__buffer_size, sp_dim))
        self.__buffer_coord = np.zeros((self.__buffer_size, 2), dtype=np.int)
        self.__buffer_index = 0

        logger.info('Going to initialize models'' Spatial Pyramid ' +
                    'representations:\n%s', ', '.join(model_list))
        for index, m_vwmat in enumerate(model_vwmat_list):
            #
            # Generate Spatial Pyramid
            #
            pyramid = patch_feat_gen.generate_representation(m_vwmat)
            self.__m_sp_mat[index, :] = pyramid




    def process_patch_representation(self, patch_feat, basename, patch_row, patch_col):  # IGNORE:unused-argument
        '''
        '''
        # Fill buffer
        if self.__buffer_index < self.__buffer_size:
            self.__buffer[self.__buffer_index, :] = patch_feat
            self.__buffer_coord[self.__buffer_index, :] = [patch_row, patch_col]
            self.__buffer_index += 1
        else:
            raise ValueError('buffer_index ran out of range!')

        # If buffer_index == buffer_size-1 fill buffer with last element
        # and compute distances
        # Note: buffer_index increment is included, before. If buffer_index
        # equals buffer_size now, it has been buffer_size-1 at the beginning
        # of this iteration (function execution)
        if self.__buffer_index == self.__buffer_size:
            self.__fill_score_mat()

    def __fill_score_mat(self):
        if self.__buffer_index == 0:
            return

        cdist_mat = cdist(self.__m_sp_mat,
                          self.__buffer[:self.__buffer_index, :],
                          metric=self.__distance_metric)
        patch_rows = self.__buffer_coord[:self.__buffer_index, 0]
        patch_cols = self.__buffer_coord[:self.__buffer_index, 1]
        for model_index, model_scores in enumerate(cdist_mat):
            self.__score_mat[patch_rows, patch_cols, model_index] = model_scores.ravel()

        self.__buffer_index = 0

    def retrieve_patches(self):
        '''
        '''

        score_filter_config = PatchScoreFilterConfig(smooth_scale=1.0,
                                                     nms_scale=2.0)
        # Empty buffer
        self.__fill_score_mat()


        ret_time = 0
        align_mat = self.__align_mat_dummy

        # Model list index must match score_mat index in 3rd dim
        for index, m_id in enumerate(self.__model_list):
            m_def = self.__doc_model_dict[m_id]
            m_bounds = m_def[2]
            m_bounds_snp = self.__patch_quant.get_patch_size_snp(m_bounds)
            m_height_snp = m_bounds_snp[1]
            height_mat = np.ones_like(align_mat) * m_height_snp
            model_score_mat = self.__score_mat[:, :, index]

            ret_time += self.__retrieval_generator.generate_retrieval_matrix(m_id, m_def,
                                                                             model_score_mat,
                                                                             align_mat,
                                                                             height_mat,
                                                                             self.__patch_yx_mat,
                                                                             score_filter_config)
        return ret_time
