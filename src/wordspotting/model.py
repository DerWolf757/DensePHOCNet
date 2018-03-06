'''
Created on Jul 10, 2013

@author: lrothack
'''
import logging, os
from esmeralda.cmd.es_hmm import MMTraining, MMInit
from esmeralda.svq import SVQWriter, SVQVectors, SVQVectorsIO
from itertools import chain
import numpy as np
import cPickle as pickle
from bofhwr.hmm.mm_reader import ConceptDefinitions

class ModelVWSerialization(object):

    def __init__(self, model_basepath):
        self.__model_basepath = model_basepath
        self.__vwlist_suffix = '.vwlist'
        self.__vwmat_suffix = '.vwmat'
        self.__svq_vectors = SVQVectors()
        self.__svq_io = SVQVectorsIO()

    def __get_model_vw_filepath(self, model_id, suffix):
        return self.__model_basepath + model_id + suffix

    def __load_vwset(self, svq_filepath):
        self.__svq_vectors.reset()
        self.__svq_io.read_vectors(svq_filepath, self.__svq_vectors)
        vwlist = []
        self.__svq_vectors.export_vectors_list(vwlist)
        vwlist = [vwitem[0] for vwitem in chain(*vwlist)]
        return set(vwlist)


    def __load_vwlist(self, svq_filepath_list):
        model_vwset = set()
        for svq_filepath in svq_filepath_list:
            vwset = self.__load_vwset(svq_filepath)
            model_vwset.update(vwset)
        model_vwlist = sorted(model_vwset)
        return model_vwlist

    def write_model_vwlist(self, model_id, svq_filepath_list):
        logger = logging.getLogger('ModelVWSerialization::write_model_vwlist')
        model_vwlist = self.__load_vwlist(svq_filepath_list)
        model_vwlist_fp = self.__get_model_vw_filepath(model_id,
                                                       self.__vwlist_suffix)
        logger.info('Writing model vwlist: %s', model_vwlist_fp)
        with open(model_vwlist_fp, 'wb') as m_vwlist_f:
            pickle.dump(model_vwlist, m_vwlist_f)

    def read_model_vwlist(self, model_id):
        logger = logging.getLogger('ModelVWSerialization::read_model_vwlist')
        model_vwlist_fp = self.__get_model_vw_filepath(model_id,
                                                       self.__vwlist_suffix)
        logger.info('Reading model vwlist: %s', model_vwlist_fp)
        with open(model_vwlist_fp, 'rb') as m_vwlist_f:
            return pickle.load(m_vwlist_f)

    def write_model_vwmat(self, model_id, model_vwmat):
        logger = logging.getLogger('ModelVWSerialization::write_model_vwmat')
        model_vwmat_fp = self.__get_model_vw_filepath(model_id,
                                                      self.__vwmat_suffix)
        logger.info('Writing model vwmat: %s', model_vwmat_fp)
        with open(model_vwmat_fp, 'wb') as m_vwmat_f:
            np.save(m_vwmat_f, model_vwmat)

    def read_model_vwmat(self, model_id):
        logger = logging.getLogger('ModelVWSerialization::read_model_vwmat')
        model_vwmat_fp = self.__get_model_vw_filepath(model_id,
                                                      self.__vwmat_suffix)
        logger.info('Reading model vwmat: %s', model_vwmat_fp)
        with open(model_vwmat_fp, 'rb') as m_vwmat_f:
            return np.load(m_vwmat_f)

class ModelSVQGenerator(object):

    def __init__(self, patch_generator, patch_quantizer,
                 model_path, svq_basepath, store_vwmat=False):
        '''
        '''
        self.__patch_generator = patch_generator
        self.__patch_quantizer = patch_quantizer
        self.__svqwriter = SVQWriter(svq_basepath)
        self.__vwmat_writer = ModelVWSerialization(model_path)
        self.__store_vwmat = store_vwmat

    def write_model_svq(self, model_dict):
        logger = logging.getLogger('ModelSVQGenerator::write_model_svq')
        # List for saving svq (filepath,seq_len) tuples
        model_svqmeta_dict = {}
        doc_model_num = len(model_dict)
        for index, (m_id, m_def) in enumerate(model_dict.iteritems()):
            logger.info('[ %04d / %04d ] going to write data for model: %s',
                        index, doc_model_num, m_id)

            # Clear list in oder to save path of next model's svq file
            self.__svqwriter.clear_svqlist()

            m_bounds = m_def[2]
            # Snap bounds to dense visual word grid
            m_bounds_snp = self.__patch_quantizer.snap_bounds_to_grid(m_bounds)
            m_tv_seq = self.__patch_generator.generate_model_termvectors(*m_bounds_snp)

            #
            # Store model svq vector sequence
            #
            self.__svqwriter.write_vectors(m_tv_seq, basename=m_id)

            # Obtain x coordinates and visual word (vw) ids, that the svq vector
            # sequence is based on
            m_xmat, m_vwmat = self.__patch_generator.generate_patch_matrices(*m_bounds_snp)

            # Store vw mat if required.
            # E.g., for further processing of the model
            if self.__store_vwmat:
                self.__vwmat_writer.write_model_vwmat(m_id, m_vwmat)

            #
            # Model meta data definitions
            #
            model_svqlist = self.__svqwriter.get_svqlist()
            model_svqpath = model_svqlist[0]
            model_svqlen = m_tv_seq.size()
            model_height_snp = m_bounds_snp[1][1] - m_bounds_snp[0][1]
            # converts x coord matrix to tuple of unique and sorted coords
            model_x_snp_tup = tuple(np.unique(m_xmat))
            model_svqmeta_dict[m_id] = (model_svqpath, model_svqlen,
                                        model_height_snp, model_x_snp_tup)

        return model_svqmeta_dict



class ModelEstimator(object):

    KEEP_TRAINING_FILES = False

    def __init__(self, model_path, vocabulary_size, train_index, model_topology,
                 model_frame_state_fun, bg_params=None, ws_models=None):
        '''
        Params:
            train_index: (Inclusive) index of the last Baum-Welch training
                iteration
            model_topology: HMM topology (Linear,Bakis)
            model_frame_state_fun_def: Linear function definition
                (two points in 4-tuple) for calculating the state number
                wrt the number of frames
        '''
        self.__model_path = model_path
        # Attention: Actual vocabulary size might differ from vocabulary size in
        # config object. Might be altered by probability term vector weighting.
        self.__vocabulary_size = vocabulary_size

        self.__train_index = train_index
        self.__model_topology = model_topology
        # Model frame state function definition
        mfsf_def = model_frame_state_fun
        mfsf_slope = ((mfsf_def[3] - mfsf_def[1]) /
                      float(mfsf_def[2] - mfsf_def[0]))
        mfsf_offset = mfsf_def[1] - (mfsf_slope * mfsf_def[0])
        mfsf_fun_min = 0.01
        self.__mfsf_fun = lambda x : max(mfsf_slope * x + mfsf_offset,
                                         mfsf_fun_min)

        self.__vwlist_generator = ModelVWSerialization(model_path)

        self.__bg_params = bg_params
        self.__ws_models = ws_models


    def init_model(self, model_id, m_transitions, m_weights):
        model_parameters = {model_id: (m_transitions, m_weights,
                                       self.__model_topology)}
        m_type = model_id + '\t' + self.__model_topology

        mminit = MMInit(self.__model_path, model_id)
        mminit.write_modelini([])
        mminit.write_modelframe([m_type])
        mminit.mm_init_with_parameters(model_parameters)


    def estimate_model(self, meta_model_dict, model_svqmeta_dict,
                       cleanup_training_files=True):
        """Builds the HMM spotting model from the ground truth

        Params:
            meta_model_dict: dictionary of meta model ids, mapping to
                a list of model ids. Each meta model will be estimated from all
                specified elementary models
            model_svqmeta_dict dictionary mapping from elementary model ids
                to a tuple of (model_svq_path,model_svqlen) where svqlen refers
                to the number of elements in the respective svq sequence.
            cleanup_training_files: Boolean flag indicating whether model
                files not needed for the specified model_state_index will be
                deleted. ATTENTION: can be overwritten with KEEP_TRAINING_FILES
                flag.
        """

        logger = logging.getLogger('ModelEstimator::estimate_model')
        meta_model_num = len(meta_model_dict)
        meta_model_dict_items = meta_model_dict.items()
        for idx, meta_model_tup in enumerate(meta_model_dict_items):
            (meta_model_id, meta_model_def) = meta_model_tup
            logger.info('[ %04d / %04d ] going to estimate model: %s',
                        idx, meta_model_num, meta_model_id)

            #
            # Model data definitions
            #

            model_align_list = []
            model_svqlen_list = []
            model_svqpath_list = []
            model_height_snp_list = []

            for m_id in meta_model_def:
                # Lookup svq meta data
                m_svq_def = model_svqmeta_dict[m_id]
                m_svqpath = m_svq_def[0]
                m_svqlen = m_svq_def[1]
                m_height_snp = m_svq_def[2]
                # Create alignment definition for sample
                m_align = meta_model_id + '[%d..%d]' % (0, m_svqlen - 1)

                # Add data to meta model lists
                model_align_list.append(m_align)
                model_svqlen_list.append(m_svqlen)
                model_svqpath_list.append(m_svqpath)
                model_height_snp_list.append(m_height_snp)

            model_svqpath_list_num = len(model_svqpath_list)
            model_id_list = [meta_model_id] * model_svqpath_list_num

            m_type = meta_model_id + '\t' + self.__model_topology

            #
            # Model initialization
            #
            mminit = MMInit(self.__model_path, meta_model_id)
            modelini_def = mminit.generate_modelini_list(model_svqpath_list,
                                                         model_align_list)
            mminit.write_modelini(modelini_def)
            mminit.write_modelframe([m_type])
            frame_state_fac_val = self.__mfsf_fun(min(model_svqlen_list))
            mminit.mm_init(self.__vocabulary_size,
                           frame_state_factor=frame_state_fac_val)

            #
            # Model training
            #
            mmtraining = MMTraining(self.__model_path, meta_model_id)
            mmtraining_def = mmtraining.generate_training_definitions_list(model_svqpath_list,
                                                                           model_id_list)
            mmtraining.write_training_definitions(mmtraining_def)
            # The training range 0..train_index will run train_index
            # iterations (0..train_index-1) and store results for
            # iteration train_index which is the model at train_index
            # --> iteration 0 updates model 0 to model 1
            mmtraining.training((0, self.__train_index))

            if self.__bg_params is not None:
                # Append background model
                logger.info('   appending additional (background) model(s): %s',
                            self.__bg_params.keys())
                mminit.append_model_definition(self.__bg_params,
                                               self.__train_index)
            if self.__ws_models is not None:
                model_height_snp_arr = np.array(model_height_snp_list)
                meta_model_height = int(np.mean(model_height_snp_arr))
                ws_params = self.__ws_models.load_ws_params(meta_model_height)
                logger.info('   appending additional (whitespace) model(s): %s',
                            ws_params.keys())
                mminit.append_model_definition(ws_params,
                                               self.__train_index)

            # Cleanup training files
            # results for training iteration N have been saved to
            # iteration id N+1
            # Example: train_index=3 results in training iterations 0..2
            # --> results for iteration 2 will be saved to iteration 3
            # --> these results are used as the basis for iteration 3
            # (which is the 4th iteration because of the 0-based index)
            if (cleanup_training_files and
                not ModelEstimator.KEEP_TRAINING_FILES):
                logger.info('Deleting model init files')
                mminit.delete_init_files()
                mmtraining.delete_training_definitions()
                logger.info('Deleting model definition files: '
                            'iterations %d to %d',
                            0, self.__train_index - 1)
                for index in range(0, self.__train_index):
                    mmtraining.delete_training_iteration(index)

            if not ModelEstimator.KEEP_TRAINING_FILES:
                logger.info('Deleting svq files (%d)', len(model_svqpath_list))
                for svq_path in model_svqpath_list:
                    os.remove(svq_path)



class VWModelGenerator(object):

    ES_MIX_PROB_THRESH = 1e-12

    def __init__(self, config, model_size):

        self.__config = config
        self.__model_size = model_size

    def generate_hmm_model_vwlist_list(self,
                                       model_list,
                                       model_concept_dict,
                                       n_patch_models):
        config = self.__config
        logger = logging.getLogger('VWModelGenerator::generate_hmm_model_vwlist_list')
        logger.info('Going to initialize model visualword lists:\n%s',
                    ', '.join(model_list))

        n_transitions = config.get_n_transitions()

        model_vw_list = []
        for m_id in model_list:
            #
            # Initialize decoders
            #
            mm_concept = model_concept_dict[m_id]
            n_states = mm_concept.n_states()
            # Export definition to numpy
            transitions_mat = np.zeros((n_states, n_transitions))
            mixtures_mat = np.zeros((n_states, self.__model_size))
            mm_concept.export_state_space(transitions_mat, mixtures_mat)
            mixtures_mat = np.array(mixtures_mat, dtype=np.float32)
            m_n_states_list = mm_concept.models_n_states()

            if n_patch_models >= ConceptDefinitions.N_MODELS_F_Q_F:
                m_ns_l = m_n_states_list[0]
                m_ns_r = m_n_states_list[-1]
                mixtures_mat = mixtures_mat[m_ns_l:-m_ns_r, :]

            if n_patch_models == ConceptDefinitions.N_MODELS_FW_Q_WF:
                m_ns_l = m_n_states_list[1]
                m_ns_r = m_n_states_list[-2]
                mixtures_mat = mixtures_mat[m_ns_l:-m_ns_r, :]

            prob_thresh = VWModelGenerator.ES_MIX_PROB_THRESH
            mixtures_mat[mixtures_mat <= prob_thresh] = 0
            # Check if there is a state with a mixture distribution
            # having positive weights for all mixtures
            # --> this is not discriminative enough for IFS decoding
            for idx, state_mix_arr in enumerate(mixtures_mat):
                state_n_mixtures = np.sum(state_mix_arr > 0)
                if state_n_mixtures == self.__model_size:
                    logger.warn('%s [state %d]:: all mixture weights positive '
                                '(non-zero): removing for IFS decoding',
                                m_id, idx)
                    mixtures_mat[idx, :] = 0

            n_states = int(mixtures_mat.shape[0])
            n_model_cells = int(config.ifs_ght_cells)
            if n_model_cells < 0:
                raise ValueError('Invalid number of GHT model cells: %d',
                                 n_model_cells)
            elif n_model_cells == 0 or n_states < n_model_cells:
                n_model_cells = n_states

            # Adjust number of model cells in case it is uneven
            # Compute offset for making number of model cells uneven
            n_cell_offset = 1 - (n_model_cells % 2)
            n_model_cells -= n_cell_offset

            # Integer division for 'regular' cell size
            s_vw_size_lr = n_states / n_model_cells
            # Center cell size is regular cell size plus the rest
            s_vw_size_center = s_vw_size_lr + (n_states % n_model_cells)
            ma_idx = 0
            # Perform integer division to obtain center index
            center_idx = n_model_cells / 2
            s_vw_list = []
            for cell_idx in range(n_model_cells):
                # Adjust cell size for regular cells or the center cell
                if cell_idx != center_idx:
                    cell_size = s_vw_size_lr
                else:
                    cell_size = s_vw_size_center
                # Obtain mixture weights (probs) for states within cell
                mixtures_c_arr = mixtures_mat[ma_idx:ma_idx + cell_size, :]
                # Sum mixture weights in order to obtain prob-voting mass
                # Results in a state-wise linear combination of mixture probs
                # and probs for the frame given the mixture.
                s_vw_arr = np.sum(mixtures_c_arr, axis=0)
                s_vw_list.append(s_vw_arr)
                ma_idx += cell_size

            # Concatenate in order to obtain prob-voting mass for all cells
            # in the model
            s_vw_arr = np.vstack(tuple(s_vw_list))
            model_vw_list.append(s_vw_arr)

        return model_vw_list


    def load_model_vwmat_list(self, model_list, doc_model_dict):
        """
        Load visual word matrices

        """
        logger = logging.getLogger('VWModelGenerator::load_model_vwmat_list')
        logger.info('Going to initialize model visualword-matrix list:\n%s',
                    ', '.join(model_list))
        model_vwmat_list = []
        for m_id in model_list:
            m_def = doc_model_dict[m_id]
            m_document = m_def[0]
            model_basepath = self.__config.get_document_model_path(m_document)
            vwlist_io = ModelVWSerialization(model_basepath)
            m_vwmat = vwlist_io.read_model_vwmat(m_id)
            model_vwmat_list.append(m_vwmat)

        return model_vwmat_list

    def load_model_vwlist_list(self, model_list, doc_model_dict):
        logger = logging.getLogger('VWModelGenerator::load_model_vwlist_list')
        logger.info('Going to load model visualword lists:\n%s',
                    ', '.join(model_list))
        models_vw_list = []
        for m_id in model_list:
            m_def = doc_model_dict[m_id]
            m_document = m_def[0]
            model_basepath = self.__config.get_document_model_path(m_document)
            vwlist_io = ModelVWSerialization(model_basepath)
            m_vwlist = vwlist_io.read_model_vwlist(m_id)
            models_vw_list.append(m_vwlist)

        return models_vw_list
