'''
Created on Jan 15, 2013

@author: lrothack
'''
import hashlib
import logging
from patrec.termvector.termvector_weighting import TermWeightingConfig
from patrec.serialization.list_io import LineListIO
import numpy as np

class WSConfig(object):
    '''
    '''

    def __init__(self, data_base_dir, tmp_base_dir, quantization_base_dir, model_base_dir,
                 decoding_mode, vt_beamwidth, vt_problow,
                 model_topology, model_frame_state_fun,
                 model_state_index, model_patch_size_reduction,
                 om_type, om_size, om_lineheights, om_termination,
                 om_detann, om_beamwidth, om_smoothing,
                 sp_partitions, sp_distance,
                 ifs_mode, ifs_ght_cells,
                 vocabulary_size, vocabulary_sampling_rate,
                 vocabulary_desc_contrast_thresh, vocabulary_mode,
                 patch_shift_denom, patch_page_thresh, patch_overlap, patch_nms,
                 patch_fitting, patch_ws_fitting,
                 patch_hyp_config, patch_hyp_region, patch_hyp_line,
                 term_weighting,
                 frame_size, frame_step, frame_dir,
                 vw_accu_filter, vw_offset, vw_grid_spacing,
                 vw_grid_bounds_mode, vw_grid_offset,
                 desc, desc_cell_size, desc_cell_struct, desc_cell_bins,
                 desc_smooth_sigma,
                 document_image_filesuffix='.png', gtp_encoding='ascii',
                 store_score_mat=False, store_retrieval_mat=True,
                 single_query_decoding=False):
        '''
        Constructor

        Params:
            data_base_dir: Path to the directory containing the George
                Washingtion dataset (input).
            tmp_base_dir: Base path for temporary data (output).
            decoding_mode: identifier for decoding the bof(-hmm) models
            vt_beamwidth: Beam width for Viterbi decoding.
            vt_problow: Floor probability for Viterbi decoding.
            quantization_base_dir: Base path for visual words (base for model).
            model_base_dir: Base path for model definitions (output).
            model_topology: HMM topology {Linear | Bakis} (es parameter).
            model_frame_state_fun: Linear function definition for calculating the number
                of model states from the number of model frames (es parameter).
                The function is specified by two points given as 4-tuple (x1,y1,x2,y2).
            model_state_index: The index indicating the number of training
                iterations during initialization or the model for decoding.
            model_patch_size_reduction: Patch size reduction per side for
                creating a more specific model
            om_type: Output model (om) identifier
            om_size: Output model size, e.g., number of densities
            om_lineheights: Tuple specifying range(start, end+1, step)
                of frame heights in pixels for extracting bof representations that
                will be modeled with the output model.
                (!) end value is inclusive (!)
            om_termination: Tuple (n_iterations, epsilon) containing
                termination criteria for estimating output model
            om_detann: Tuple containing deterministic annealing schedule.
            om_beamwidth: Tuple containing beam width for output model estimation
                and decoding.
            om_smoothing: Tuple containing output model specific smoothing parameters
            sp_partitions: Defining Spatial Pyramid partition layout, given as
                tuple of tuples. One tuple per layer specifying cell rows and cols.
            sp_distance: Spatial Pyramid distance metric identifier given as string.
            ifs_mode: Switch between visual word and frame indexing {'vw', 'frame'}
            ifs_ght_cells: Number of cells in the query model for GHT
                (generalized Hough transform) IFS decoding
            vocabulary_size: Size / dimension of the visual vocabulary obtained from
                clustering --> for interpreting term vectors in terms
                of probability distributions vocabulary_size will be increased
                by one. See TermProbabilityDistributionWeighting (adjusted vocabulary_size).
            vocabulary_sampling_rate: Percentage in ]0,1] of descriptors used for clustering
            vocabulary_desc_contrast_thresh: Thresholding based in SIFT contrast
                0: disable
            vocabulary_mode: Clustering mode {'lloyd', 'mqueen'}
                in order to choose Lloyd's or MacQueen's algorithm, respectively
            patch_shift_denom: Denominator (1/patch_shift_denom) for determining the
                horizontal and vertical patch shift with respect to the patch size.
            patch_page_thresh: Number of patch representations considered per page/document.
            patch_nms: Non-minimum-suppression parameter specifying the filter size (2-tuple).
            patch_fitting: Parameter defining patch to word occurrence fitting
            patch_ws_fitting: Whitespace model configuration for patch fitting
            patch_hyp_config: CCSpaceConfig object defining patch region
                hypotheses generation, optional: None
            patch_hyp_region: Parameters for fitting patches to hypotheses
                Tuple (beam_factor, weighting_factor [distance <--> size])
            patch_hyp_line: Controls line hypotheses filtering
                Tuple (beam_factor, min, step)
            term_weighting: String specifying the term weighting mode
                {'', 'bin','idf', 'bin-idf'}
            frame_size: Size of the sliding window.
            frame_step: Shift of the sliding window.
            frame_dir: Shift direction of the sliding window.
            vw_accu_filter: Accumulator filter element
            vw_offset: Offset to be added to visual word indices for correcting
                their range ([0..N-1])
            vw_grid_spacing: The feature spacing in the dense grid.
            vw_grid_bounds_mode: {img|gt} img: use image bounds, gt: obtain bounds from ground truth
            vw_grid_offset: Coordinates offset for the vw grid in the image in
                (x, y) coordinates.
            desc: Descriptor type identifier. Internally supported: lgh, sift
                Can be used for integrating external features by specifying any
                identifier and providing features in the correct format at the
                correct filepath.
            desc_cell_size: The area covered by a descriptor cell.
            desc_cell_struct: The descriptor cell layout as tuple in (row,col).
                Not supported for desc=='sift' --> is 4x4 by default
            desc_cell_bins: Number of bins (dimensions) per descriptor cell.
            desc_smooth_sigma: Standard deviation for smoothing image (Gaussian kernel)
                before descriptor extraction.
            document_image_filesuffix: Document image filename ending
            gtp_encoding: File encoding for ground truth files, default: ascii
            store_score_mat: True / False controlling score matrix storage
            store_score_mat: True / False controlling retrieval matrix storage
            single_query_decoding: True / False indicating if multiple queries
                of the same size are decoded separately or together
                (patch representations only computed once)

        '''

        self.__data_base = data_base_dir
        self.__tmp_base = tmp_base_dir
        self.__quantization_base = quantization_base_dir
        self.__model_base = model_base_dir

        self.__decoding_mode = decoding_mode
        self.__vt_beamwidth = vt_beamwidth
        self.__vt_problow = vt_problow
        self.__vocabulary_size = vocabulary_size
        self.__vocabulary_sampling_rate = vocabulary_sampling_rate
        self.__vocabulary_desc_contrast_thresh = vocabulary_desc_contrast_thresh
        self.__vocabulary_mode = vocabulary_mode
        self.__patch_page_thresh = patch_page_thresh
        self.__om_type = om_type
        self.__om_size = om_size
        self.__om_lineheights = om_lineheights
        self.__om_termination = om_termination
        self.__om_detann = om_detann
        self.__om_beamwidth = om_beamwidth
        self.__om_smoothing = om_smoothing
        self.__sp_partitions = sp_partitions
        self.__sp_distance = sp_distance
        self.__ifs_mode = ifs_mode
        self.__ifs_frame_caching = False
        self.__ifs_ght_cells = ifs_ght_cells
        self.__model_topology = model_topology
        self.__model_frame_state_fun = model_frame_state_fun
        self.__model_state_index = model_state_index
        self.__model_patch_size_reduction = model_patch_size_reduction
        self.__patch_shift_denom = patch_shift_denom
        self.__patch_overlap = patch_overlap
        self.__patch_nms = patch_nms
        self.__patch_fitting = patch_fitting
        self.__patch_ws_fitting = patch_ws_fitting
        self.__patch_hyp_config = patch_hyp_config
        self.__patch_hyp_region = patch_hyp_region
        self.__patch_hyp_line = patch_hyp_line
        self.__term_weighting = term_weighting
        self.__frame_size = frame_size
        self.__frame_step = frame_step
        self.__frame_dir = frame_dir
        self.__vw_accu_filter = vw_accu_filter
        self.__vw_offset = vw_offset
        self.__vw_grid_spacing = vw_grid_spacing
        self.__vw_grid_bounds_mode = vw_grid_bounds_mode
        self.__vw_grid_offset = vw_grid_offset
        self.__desc_cell_size = desc_cell_size
        self.__desc_cell_struct = desc_cell_struct
        self.__desc_cell_bins = desc_cell_bins
        self.__desc_smooth_sigma = desc_smooth_sigma
        self.__desc = desc
        self.__store_score_mat = store_score_mat
        self.__store_retrieval_mat = store_retrieval_mat
        self.__document_image_filesuffix = document_image_filesuffix
        self.__gtp_encoding = gtp_encoding
        self.__single_query_decoding = single_query_decoding

        path_tup = self.__generate_paths()
        self.__feat_path = path_tup[0]
        self.__svq_path = path_tup[1]
        self.__quantization_path = path_tup[2]
        self.__patches_path = path_tup[3]
        self.__om_path = path_tup[4]
        self.__model_path = path_tup[5]
        self.__scores_path = path_tup[6]


    def print_ws_config(self):
        ep_lineheights_spec = ','.join('%d' % l
                                        for l in self.__om_lineheights)
        ep_termination_spec = ','.join('%.1e' % t
                                        for t in self.__om_termination)
        ep_detann_spec = ','.join('%g' % t
                                   for t in self.__om_detann)
        patch_hyp_spec = (self.__patch_hyp_config.generate_specifier()
                          if self.has_patch_hyp_config() else '')
        ws_config_desc_list = [
        '\n',
        'data_base: %s' % self.__data_base,
        'tmp_base: %s' % self.__tmp_base,
        'model_base: %s' % self.__model_base,
        'quantization_base: %s' % self.__quantization_base,
        'decoding_mode: %s' % self.__decoding_mode,
        'vt_beamwidth: %d' % self.__vt_beamwidth,
        'vt_problow: %g' % self.__vt_problow,
        'model_topology: %s' % self.__model_topology,
        'model_frame_state_fun: (%.2f, %.2f, %.2f, %.2f)' % self.__model_frame_state_fun,
        'model_state_index: %d' % self.__model_state_index,
        'model_patch_size_reduction: %d-%d-%d-%d' % self.__model_patch_size_reduction,
        'om_type: %s' % self.__om_type,
        'om_size: %d' % self.__om_size,
        'om_lineheights: (%s)' % ep_lineheights_spec,
        'om_termination: (%s)' % ep_termination_spec,
        'om_detann: (%s)' % ep_detann_spec,
        'om_beamwidth: (%g, %g)' % self.__om_beamwidth,
        'om_smoothing: %s' % str(self.__om_smoothing),
        'sp_partitions: %s' % str(self.__sp_partitions),
        'sp_distance: %s' % self.__sp_distance,
        'ifs_mode: %s' % self.__ifs_mode,
        'ifs_ght_cells: %d' % self.__ifs_ght_cells,
        'vocabulary_size: %d' % self.__vocabulary_size,
        'vocabulary_sampling_rate: %g' % self.__vocabulary_sampling_rate,
        'vocabulary_desc_contrast_thresh: %g' % self.__vocabulary_desc_contrast_thresh,
        'vocabulary_mode: %s' % self.__vocabulary_mode,
        'patch_shift_denom: %d' % self.__patch_shift_denom,
        'patch_overlap: %g' % self.__patch_overlap,
        'patch_nms: (%d, %d)' % self.__patch_nms,
        'patch_fitting: %s' % str(self.__patch_fitting),
        'patch_ws_fitting: %s' % str(self.__patch_ws_fitting),
        'patch_hyp_config: %s' % patch_hyp_spec,
        'patch_hyp_region: %s' % str(self.__patch_hyp_region),
        'patch_hyp_line: %s' % str(self.__patch_hyp_line),
        'patch_page_thresh: %d' % self.__patch_page_thresh,
        'term_weighting: %s' % str(self.__term_weighting),
        'vw_accu_filter: (%d, %d)' % self.__vw_accu_filter,
        'vw_offset: %d' % self.__vw_offset,
        'vw_grid_spacing: (%d, %d)' % self.__vw_grid_spacing,
        'vw_grid_bounds_mode: %s' % self.__vw_grid_bounds_mode,
        'vw_grid_offset: (%d, %d)' % self.__vw_grid_offset,
        'desc: %s' % self.__desc,
        'desc_cell_size: %d' % self.__desc_cell_size,
        'desc_cell_struct: (%d, %d)' % self.__desc_cell_struct,
        'desc_cell_bins: %d' % self.__desc_cell_bins,
        'desc_smooth_sigma: %.1f' % self.__desc_smooth_sigma,
        'document_image_filesuffix: %s' % self.__document_image_filesuffix,
        'store_score_mat: %s' % self.__store_score_mat,
        'store_retrieval_mat: %s' % self.__store_retrieval_mat,
        'single_query_decoding: %s' % self.__single_query_decoding,
        '\n']

        logger = logging.getLogger('WSConfig::INFO')
        logger.info('\n'.join(ws_config_desc_list))

        self.print_ws_config_id()

    def ws_config_id(self):
        """Generate and return short config description.

        ID consists of config directory names that contain important params
        """
        dirname_tup = self._generate_dirnames()
        quantization_dirname = dirname_tup[0]
        patches_dirname = dirname_tup[1]
        om_dirname = dirname_tup[2]
        models_dirname = dirname_tup[3]
        scores_dirname = dirname_tup[4]

        config_id = '\n'.join(['\nQuantization:\t %s' % quantization_dirname,
                   'Patches:\t %s' % patches_dirname,
                   'Density model:\t %s' % om_dirname,
                   'Experiment:\t %s' % models_dirname,
                   'Retrieval:\t %s' % scores_dirname, ])
        return config_id

    def print_ws_config_id(self):
        """Print config ID generated by ws_config_id method
        """
        ws_config_id = self.ws_config_id()
        logger = logging.getLogger('WSConfig::ID')
        logger.info(''.join(['\n', ws_config_id, '\n']))

    def hash_digest(self, hash_algo=None):
        """Generate and return hash digest computed over all private class
        member attributes (class member variables starting with __ )

        Params:
            hash_algo: string identifying the hashing algorithms. The algorithm
                must be supported by the hashlib library. Default: md5
        """
        if hash_algo is None:
            hash_algo = 'md5'
        if hash_algo not in hashlib.algorithms_guaranteed:
            raise ValueError('Unsupported hash algorithm: %s' % hash_algo)
        hash_val = hashlib.new(hash_algo)
        clazz_name = self.__class__.__name__
        attributes_private = [attr for attr in dir(self)
                              if attr.startswith('_%s__' % clazz_name)]
        for attr in attributes_private:
            attr_value = getattr(self, attr)
            hash_val.update(str(attr_value))

        return hash_val.digest()

    def _generate_dirnames(self):
        quantization_dirname = self.__get_quantization_specifier()
        patch_spec = self.__get_patch_specifier()
        term_spec = self.__get_term_specifier()
        frame_spec = self.__get_frame_specifier()
        patches_dirname = '%s_%s%s' % (patch_spec, term_spec, frame_spec)
        om_spec = self.__get_om_model_specifier()
        om_dirname = om_spec if self.has_om_model() else ''
        m_topology_spec = 'hmm-%s' % self.__model_topology
        m_frame_sf_spec = '-fsf%.2f-%.2f-%.2f-%.2f' % self.__model_frame_state_fun
        m_patch_sr_spec = '-psr%d-%d-%d-%d' % self.__model_patch_size_reduction
        m_ws_height_spec = self.__get_ws_height_specifier()
        models_dirname = ''.join([m_topology_spec,
                                  m_frame_sf_spec,
                                  m_patch_sr_spec,
                                  m_ws_height_spec])
        decoding_mode_spec = self.__get_decoding_mode_specifier()
        patch_fitting_spec = self.__get_patch_fitting_specifier()
        patch_hyp_spec = self.__get_patch_hyp_specifier()
        s_param_tup = (decoding_mode_spec,
                       self.__patch_page_thresh,
                       self.__patch_overlap,
                       self.__patch_nms[0],
                       self.__patch_shift_denom,
                       patch_fitting_spec,
                       patch_hyp_spec)
        scores_dirname = 'scores_eval-%s-pt%d-po%g-pnms%d-psd%d%s%s' % s_param_tup
        dirname_tup = (quantization_dirname,
                       patches_dirname,
                       om_dirname,
                       models_dirname,
                       scores_dirname)
        return dirname_tup

    def __get_quantization_specifier(self):
        desc_spec = self.__get_desc_specifier()
        voc_specifier = self.__get_voc_specifier()
        quantization_spec = '%s_%s' % (desc_spec, voc_specifier)
        return quantization_spec

    def __get_term_specifier(self):
        term_spec = ('term-%s_' % self.__term_weighting
                     if self.has_term_weighting() else '')
        return term_spec

    def __get_patch_specifier(self):
        if self.use_patch_hyp_config():
            patch_qualifer = self.__patch_hyp_config.generate_specifier()
        else:
            patch_qualifer = 'sgq'

        patch_spec = 'patch-%s' % patch_qualifer

        return patch_spec

    def __get_frame_specifier(self):
        frame_spec = 'frame-box-%d-%d-%s' % (self.__frame_size,
                                             self.__frame_step,
                                             self.__frame_dir)
        return frame_spec

    def __get_patch_fitting_specifier(self):
        patch_fitting_specifier = ''
        if self.has_patch_fitting():
            patch_fitting_mode = self.__patch_fitting[0]
            patch_width_expansion = self.__patch_fitting[1]
            patch_fitting_specifier = '-pfit_%s_pwe%g' % (patch_fitting_mode,
                                                          patch_width_expansion)
        return patch_fitting_specifier

    def __get_patch_hyp_specifier(self):
        patch_hyp_spec = ''

        if self.has_patch_region_hyp_config():
            patch_hyp_spec += '-p_hyp_r%g-%g' % self.__patch_hyp_region[1:]

        if self.has_patch_line_hyp_config():
            patch_hyp_spec += '-p_hyp_l%g-%g-%g' % self.__patch_hyp_line[1:]

        return patch_hyp_spec

    def __get_decoding_mode_specifier(self):
        decoding_params_spec = ''
        if 'vt' in self.__decoding_mode:
            decoding_params_spec = '-b%d-f%g' % (self.__vt_beamwidth,
                                                 self.__vt_problow)
        elif 'sp' in self.__decoding_mode:
            param_list = list(np.array(self.__sp_partitions).ravel())
            param_list.append(self.__sp_distance)
            decoding_params_spec = '-pl%d-%d-%d-%d-d-%s' % tuple(param_list)

        if 'ifs' in self.__decoding_mode:
            ifs_param_tup = (self.__ifs_mode,
                             self.__ifs_ght_cells,
                             self.__vw_accu_filter[0],
                             self.__vw_accu_filter[1])

            decoding_params_spec += '-%s-c%d-af%d-%d' % ifs_param_tup

        return self.__decoding_mode + decoding_params_spec

    def __get_ws_height_specifier(self):
        ws_height_spec = ''
        # If patch hypotheses are available white space models are expected
        # to be estimated by default!
        if self.has_patch_hyp_config():
            ws_height_spec = '-wsh%d-%d-%d' % self.__patch_ws_fitting
        return ws_height_spec

    def __get_om_model_specifier(self):
        om_model_spec = ''
        if self.has_om_model():
            lineheights_spec = '-'.join('%d' % l
                                         for l in self.__om_lineheights)
            smoothing_spec = '-'.join(str(p) for p in self.__om_smoothing)
            detann_spec = 'da' + '-'.join('%g' % da for da in self.__om_detann)
            term_spec = 't' + '-'.join('%g' % t for t in self.__om_termination)
            om_model_spec = '%s%d_%s_%s_%s_%s' % (self.om_type,
                                                  self.om_size,
                                                  lineheights_spec,
                                                  smoothing_spec,
                                                  detann_spec,
                                                  term_spec)
        else:
            # Note: Will *not* be used in bof-hmm directory structure
            # --> backward compatibility
            # also see self._generate_dirname()
            # and self.get_feature_export_path()
            om_model_spec = 'bof'
        return om_model_spec

    def __get_desc_specifier(self):
        if self.__vw_grid_spacing[0] != self.__vw_grid_spacing[1]:
            raise ValueError('Error: Only regular grid spacing supported for ' +
                             'vw_grid_spacing parameter encoding')
        desc_spec = ''
        if self.__desc == 'sift' and self.desc_cell_struct == (4, 4):
            desc_size = self.__desc_cell_size * 4

            param_tup = (self.__vw_grid_spacing[0],
                         self.__desc,
                         self.__desc_smooth_sigma,
                         desc_size)
            desc_spec = 'dense-rg%d_%s-sm%.1f-sv%d' % param_tup

        else:
            param_tup = (self.__vw_grid_spacing[0],
                         self.__desc,
                         self.__desc_smooth_sigma,
                         self.__desc_cell_size,
                         self.__desc_cell_struct[0],
                         self.__desc_cell_struct[1])

            desc_spec = 'dense-rg%d_%s-sm%.1f-sv%d-c%d-%d' % param_tup

            if self.__desc != 'sift':
                desc_spec = '%s-%d' % (desc_spec, self.__desc_cell_bins)

        return desc_spec


    def __get_voc_specifier(self):
        sampling_specifier = 'sr'
        param_tup = (self.__vocabulary_mode,
                     self.__vocabulary_size,
                     sampling_specifier,
                     self.__vocabulary_sampling_rate,
                     self.__vocabulary_desc_contrast_thresh)

        voc_specifier = '%s-hq%d-%s%g-ct%g' % param_tup



        return voc_specifier

    def __generate_paths(self):
        dirname_tup = self._generate_dirnames()
        quantization_dirname = dirname_tup[0]
        patches_dirname = dirname_tup[1]
        om_dirname = dirname_tup[2]
        models_dirname = dirname_tup[3]
        scores_dirname = dirname_tup[4]

        patches_dir_suf = quantization_dirname + '/' + patches_dirname
        if self.has_om_model():
            om_dir_suf = patches_dir_suf + '/' + om_dirname
            models_dir_suf = om_dir_suf + '/' + models_dirname
        else:
            om_dir_suf = ''
            models_dir_suf = patches_dir_suf + '/' + models_dirname

        scores_dir_suf = models_dir_suf + '/' + scores_dirname

        feat_path = self.__tmp_base + quantization_dirname + '/'
        svq_path = self.__tmp_base + models_dir_suf + '/'
        quantization_path = self.__quantization_base + quantization_dirname + '/'
        patches_path = self.__quantization_base + patches_dir_suf + '/'
        om_path = self.__model_base + om_dir_suf + '/'
        model_path = self.__model_base + models_dir_suf + '/'
        scores_path = self.__model_base + scores_dir_suf + '/'

        path_tup = (feat_path,
                    svq_path,
                    quantization_path,
                    patches_path,
                    om_path,
                    model_path,
                    scores_path)
        return path_tup

    def _update_paths(self):
        path_tup = self.__generate_paths()
        self.__feat_path = path_tup[0]
        self.__svq_path = path_tup[1]
        self.__quantization_path = path_tup[2]
        self.__patches_path = path_tup[3]
        self.__om_path = path_tup[4]
        self.__model_path = path_tup[5]
        self.__scores_path = path_tup[6]

    def get_tmp_dir(self):
        return self.__tmp_base

    def get_decoding_mode(self):
        return self.__decoding_mode

    def get_vt_beamwidth(self):
        return self.__vt_beamwidth

    def get_vt_problow(self):
        return self.__vt_problow

    def use_vt_decoding(self):
        return True if 'vt' in self.__decoding_mode else False

    def use_ifs_decoding(self):
        return True if 'ifs' in self.__decoding_mode else False

    def use_sp_decoding(self):
        return True if self.__decoding_mode == 'sp' else False

    def get_patch_shift_denom(self):
        return self.__patch_shift_denom

    def get_patch_overlap(self):
        return self.__patch_overlap

    def get_patch_nms(self):
        return self.__patch_nms

    def get_patch_fitting(self):
        return self.__patch_fitting

    def get_patch_ws_fitting(self):
        return self.__patch_ws_fitting

    def has_patch_fitting(self):
        return (len(self.__patch_fitting) > 1 and
                self.__patch_fitting[0] != '' and
                'vt' in self.__decoding_mode)

    def has_bg_fitting(self):
        return (self.has_patch_fitting() and
                'bg' in self.__patch_fitting[0])

    def has_bg_normalization(self):
        return (self.has_bg_fitting() and
                self.has_whitespace_fitting() and
                'norm' in self.__patch_fitting[0])

    def has_whitespace_fitting(self):
        return (self.has_patch_hyp_config and
                self.has_patch_fitting() and
                'ws' in self.__patch_fitting[0])

    def has_term_weighting(self):
        return self.__term_weighting is not None and self.__term_weighting != ''

    def has_bin_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'bin'

    def has_idf_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'idf'

    def has_binidf_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'bin-idf'

    def has_vwc_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'vwc'

    def has_binvwc_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'bin-vwc'

    def has_binvwcidf_term_weighting(self):
        return self.has_term_weighting() and self.__term_weighting == 'bin-vwc-idf'

    def get_patch_hyp_config(self):
        return self.__patch_hyp_config

    def get_patch_hyp_region(self):
        return self.__patch_hyp_region

    def get_patch_hyp_line(self):
        return self.__patch_hyp_line

    def get_ws_context_id(self):
        return 'ws-%d-%d-%d' % self.__patch_ws_fitting

    def get_term_weighting(self):
        return self.__term_weighting

    def get_frame_size(self):
        return self.__frame_size

    def get_frame_step(self):
        return self.__frame_step

    def get_frame_dir(self):
        return self.__frame_dir

    def get_vw_accu_filter(self):
        return self.__vw_accu_filter

    def get_vw_offset(self):
        return self.__vw_offset

    def get_vw_grid_spacing(self):
        return self.__vw_grid_spacing

    def get_vw_grid_bounds_mode(self):
        return self.__vw_grid_bounds_mode

    def get_vw_grid_offset(self):
        return self.__vw_grid_offset

    def get_desc(self):
        return self.__desc

    def get_desc_cell_size(self):
        return self.__desc_cell_size

    def get_desc_cell_struct(self):
        return self.__desc_cell_struct

    def get_desc_cell_bins(self):
        return self.__desc_cell_bins

    def get_desc_smooth_sigma(self):
        return self.__desc_smooth_sigma

    def is_store_score_mat(self):
        return self.__store_score_mat

    def is_store_retrieval_mat(self):
        return self.__store_retrieval_mat

    def is_single_query_decoding(self):
        return self.__single_query_decoding

    def get_scores_path(self, document_id=None):
        if document_id is None:
            return self.__scores_path
        else:
            return self.__scores_path + document_id + '/'

    def get_feature_export_specifiers(self):
        spec_tup = (self.__get_quantization_specifier(),
                    self.__get_frame_specifier(),
                    self.__get_om_model_specifier())
        return spec_tup

    def get_feature_export_path(self, path_qualifier='svqx'):
        feature_specifiers = self.get_feature_export_specifiers()
        output_path = ''.join((self.__feat_path,
                               feature_specifiers[1], '/',
                               feature_specifiers[2], '/',
                               path_qualifier, '/'))
        return output_path

    def get_svq_path(self, document_id):
        return self.__svq_path + document_id + '/svqx/'

    def get_aligntmp_path(self, document_id):
        return self.__svq_path + document_id + '/align/'

    def set_document_image_filesuffix(self, file_suffix):
        self.__document_image_filesuffix = file_suffix

    def get_gtp_encoding(self):
        return self.__gtp_encoding

    def set_gtp_encoding(self, encoding):
        self.__gtp_encoding = encoding

    def get_document_image_filepath(self, document_id):
        return ''.join((self.data_base,
                       'pages/',
                       document_id,
                       self.__document_image_filesuffix))

    def get_svq_list_filepath(self, document_id, svq_list_id):
        return self.get_svq_path(document_id) + svq_list_id + '.lst'

    def __get_align_msi_id(self):
        align_msi_id = 'align-msi%02d' % self.__model_state_index
        return align_msi_id

    def get_score_mat_filepath(self, document_id, query_id):
        align_msi_id = self.__get_align_msi_id()
        return '%s%s_%s_%s.scores.npz' % (self.get_scores_path(document_id),
                                          query_id,
                                          align_msi_id,
                                          document_id)

    def get_retrieval_time_filepath(self, document_id, eval_id=None):
        eval_id = self.__get_eval_id_specifier(eval_id)
        align_msi_id = self.__get_align_msi_id()
        if self.__single_query_decoding:
            retrieval_time_fileid = 'single_query'
        else:
            retrieval_time_fileid = 'multi_query'
        return '%s%s%s_%s_time.txt' % (self.get_scores_path(document_id),
                                       eval_id,
                                       retrieval_time_fileid,
                                       align_msi_id)

    def get_retrieval_mat_filepath(self, document_id, query_id):
        align_msi_id = self.__get_align_msi_id()
        return '%s%s_%s_%s.retrieval.bin' % (self.get_scores_path(document_id),
                                             query_id,
                                             align_msi_id,
                                             document_id)


    def get_query_results_list_filepath(self, score_thresh_factor, eval_id=None):
        result_type_id = 'queryresults'
        file_suffix = '.lst'
        eval_score_thresh_id = '_retrieval-stf%.2f' % score_thresh_factor
        return self.__get_results_list_filepath(result_type_id,
                                                file_suffix,
                                                eval_id,
                                                eval_spec=eval_score_thresh_id)
    def get_query_results_export_filepath(self, score_thresh_factor, eval_id=None):
        result_type_id = 'queryresults_export'
        file_suffix = '.p'
        eval_score_thresh_id = '_retrieval-stf%.2f' % score_thresh_factor
        return self.__get_results_list_filepath(result_type_id,
                                                file_suffix,
                                                eval_id,
                                                eval_spec=eval_score_thresh_id)

    def get_eval_results_filepath(self, score_thresh_factor, eval_id=None):
        result_type_id = 'eval_summary'
        file_suffix = '.txt'
        eval_score_thresh_id = '_retrieval-stf%.2f' % score_thresh_factor
        return self.__get_results_list_filepath(result_type_id,
                                                file_suffix,
                                                eval_id,
                                                eval_spec=eval_score_thresh_id)

    def get_time_results_filepath(self, eval_id=None):
        result_type_id = 'time_summary'
        file_suffix = '.txt'
        return self.__get_results_list_filepath(result_type_id,
                                                file_suffix,
                                                eval_id,
                                                eval_spec='')

    def get_n_transitions(self):
        model_topology = self.model_topology
        if model_topology == 'Linear':
            n_transitions = 2
        elif model_topology == 'Bakis':
            n_transitions = 3
        else:
            raise ValueError('Unsupported model topology: %s' % model_topology)
        return n_transitions

    def __get_results_list_filepath(self, result_type_id,
                                    file_suffix, eval_id,
                                    eval_spec):
        eval_id = self.__get_eval_id_specifier(eval_id)
        align_msi_id = self.__get_align_msi_id()
        return '%s%s_%s%s%s%s' % (self.__scores_path,
                                  result_type_id,
                                  eval_id,
                                  align_msi_id,
                                  eval_spec,
                                  file_suffix)

    @staticmethod
    def __get_eval_id_specifier(eval_id):
        if eval_id is None:
            eval_id = ''
        else:
            eval_id = eval_id + '_'
        return eval_id

    def get_quantization_path(self):
        return self.__quantization_path

    def get_patches_path(self):
        return self.__patches_path

    def has_patch_hyp_config(self):
        return self.__patch_hyp_config is not None

    def use_patch_hyp_config(self):
        return (self.has_patch_region_hyp_config() or
                self.has_patch_line_hyp_config() or
                self.has_whitespace_fitting())

    def has_patch_region_hyp_config(self):
        return (self.__patch_hyp_config is not None and
                type(self.__patch_hyp_region) is tuple and
                len(self.__patch_hyp_region) > 1 and
                self.__patch_hyp_region[0] == 'reg')

    def has_patch_line_hyp_config(self):
        return (self.__patch_hyp_config is not None and
                type(self.__patch_hyp_line) is tuple and
                len(self.__patch_hyp_line) > 1 and
                self.__patch_hyp_line[0] == 'line')

    def has_om_model(self):
        return (self.om_type == 'ep' or
                self.om_type == 'vmf' or
                self.om_type == 'unigram' or
                self.om_type == 'mult')

    def model_size(self):
        if self.has_om_model():
            return self.om_size
        else:
            term_norm = self.get_term_normalization()
            model_size = term_norm.final_termvectors_dim()
            return model_size

    def get_om_model_path(self):
        om_type = self.om_type
        return ''.join([self.__om_path, '%s_model/' % om_type])

    def get_om_beamwidth(self):
        return self.__om_beamwidth

    def get_om_smooting(self):
        self.__check_smooting_params(self.__om_smoothing[0])
        return self.__om_smoothing

    def __check_smooting_params(self, om_smoothing_type):
        valid_smoothing_type = True

        if self.om_type == 'ep' or self.om_type == 'vmf':
            valid_smoothing_type = om_smoothing_type == 'dyn'
        elif self.om_type == 'mult' or self.om_type == 'unigram':
            valid_smoothing_type = (om_smoothing_type == 'bg' or
                                    om_smoothing_type == 'unf')
        if not valid_smoothing_type:
            raise ValueError('Unsupported smoothing type for ' +
                             'output model: %s' % om_smoothing_type)

    def get_om_model_filepaths(self, model_id=''):
        model_path = self.get_om_model_path()
        if self.om_type == 'ep':
            return self.__get_ep_model_filepaths(model_path, model_id)
        elif self.om_type == 'vmf':
            return self.__get_vmf_model_filepaths(model_path, model_id)
        elif self.om_type == 'mult':
            return self.__get_mult_model_filepaths('mult', model_path, model_id)
        elif self.om_type == 'unigram':
            return self.__get_mult_model_filepaths('unigram', model_path,
                                                   model_id)
        else:
            raise ValueError('Unsupported output model type')

    @staticmethod
    def __get_vmf_model_filepaths(model_path, model_id):
        mixture_filename = 'vmf_mixture_arr%s.npy' % model_id
        mixture_filepath = model_path + mixture_filename
        mu_filename = 'vmf_mu_arr%s.npy' % model_id
        mu_filepath = model_path + mu_filename
        kappa_filename = 'vmf_kappa_arr%s.npy' % model_id
        kappa_filepath = model_path + kappa_filename

        return mixture_filepath, mu_filepath, kappa_filepath

    @staticmethod
    def __get_ep_model_filepaths(model_path, model_id):
        mixture_filename = 'ep_mixture_mat%s.npy' % model_id
        mixture_filepath = model_path + mixture_filename
        beta_filename = 'ep_beta_mat%s.npy' % model_id
        beta_filepath = model_path + beta_filename

        return mixture_filepath, beta_filepath

    @staticmethod
    def __get_mult_model_filepaths(mult_model, model_path, model_id):
        mixture_filename = '%s_mixture_mat%s.npy' % (mult_model, model_id)
        mixture_filepath = model_path + mixture_filename
        prob_filename = '%s_prob_mat%s.npy' % (mult_model, model_id)
        prob_filepath = model_path + prob_filename
        smooth_filename = '%s_smooth_mat%s.npy' % (mult_model, model_id)
        smooth_filepath = model_path + smooth_filename

        return mixture_filepath, prob_filepath, smooth_filepath

    def get_document_model_path(self, document_id):
        return self.__model_path + document_id + '/'

    def get_data_base(self):
        return self.__data_base

    def get_patch_page_thresh(self):
        return self.__patch_page_thresh

    def get_model_topology(self):
        return self.__model_topology

    def get_model_frame_state_fun(self):
        return self.__model_frame_state_fun

    def get_model_state_index(self):
        return self.__model_state_index

    def get_model_patch_size_reduction(self):
        return self.__model_patch_size_reduction

    def get_om_type(self):
        return self.__om_type

    def get_om_size(self):
        return self.__om_size

    def get_om_lineheights(self):
        return self.__om_lineheights

    def get_om_termination(self):
        return self.__om_termination

    def get_om_detann(self):
        return self.__om_detann

    def get_sp_partitions(self):
        return self.__sp_partitions

    def get_sp_distance(self):
        return self.__sp_distance

    def get_ifs_mode(self):
        return self.__ifs_mode

    def use_ifs_frame_caching(self):
        return self.__ifs_frame_caching

    def use_ifs_frame_indexing(self):
        vw_mode = True if self.__ifs_mode == 'frame' else False
        return self.use_ifs_decoding() and vw_mode

    def use_ifs_vw_indexing(self):
        vw_mode = True if self.__ifs_mode == 'vw' else False
        return self.use_ifs_decoding() and vw_mode

    def get_ifs_ght_cells(self):
        return self.__ifs_ght_cells

    def get_vocabulary_size(self):
        return self.__vocabulary_size

    def get_vocabulary_sampling_rate(self):
        return self.__vocabulary_sampling_rate

    def get_vocabulary_desc_contrast_thresh(self):
        return self.__vocabulary_desc_contrast_thresh

    def get_vocabulary_mode(self):
        return self.__vocabulary_mode

    def use_lloyd_clustering(self):
        return True if self.vocabulary_mode == 'lloyd' else False

    def use_mqueen_clustering(self):
        return True if self.vocabulary_mode == 'mqueen' else False

    def use_mqueenpp_clustering(self):
        return True if self.vocabulary_mode == 'mqueenpp' else False

    def get_term_normalization(self):
        return TermWeightingConfig.get_probability_weighting(self.__vocabulary_size)

    def set_decoding_mode(self, decoding_mode):
        self.__decoding_mode = decoding_mode
        self._update_paths()

    def set_vt_beamwidth(self, vt_beamwidth):
        self.__vt_beamwidth = vt_beamwidth
        self._update_paths()

    def set_vt_problow(self, vt_problow):
        self.__vt_problow = vt_problow
        self._update_paths()

    def set_patch_shift_denom(self, patch_shift_denom):
        self.__patch_shift_denom = patch_shift_denom
        self._update_paths()

    def set_patch_overlap(self, patch_overlap):
        self.__patch_overlap = patch_overlap
        self._update_paths()

    def set_patch_nms(self, patch_nms):
        self.__patch_nms = patch_nms
        self._update_paths()

    def set_patch_fitting(self, patch_fitting):
        self.__patch_fitting = patch_fitting
        self._update_paths()

    def set_patch_ws_fitting(self, patch_ws_fitting):
        self.__patch_ws_fitting = patch_ws_fitting
        self._update_paths()


    def set_patch_hyp_config(self, patch_hyp_config):
        self.__patch_hyp_config = patch_hyp_config
        self._update_paths()

    def set_patch_hyp_region(self, patch_hyp_region):
        self.__patch_hyp_region = patch_hyp_region
        self._update_paths()

    def set_patch_hyp_line(self, patch_hyp_line):
        self.__patch_hyp_line = patch_hyp_line
        self._update_paths()

    def set_term_weighting(self, term_weighting):
        self.__term_weighting = term_weighting
        self._update_paths()

    def set_frame_size(self, frame_size):
        self.__frame_size = frame_size
        self._update_paths()

    def set_frame_step(self, frame_step):
        self.__frame_step = frame_step
        self._update_paths()

    def set_vw_accu_filter(self, vw_accu_filter):
        self.__vw_accu_filter = vw_accu_filter
        self._update_paths()

    def set_frame_dir(self, frame_dir):
        self.__frame_dir = frame_dir
        self._update_paths()

    def set_vw_offset(self, vw_offset):
        self.__vw_offset = vw_offset

    def set_vw_grid_spacing(self, vw_grid_spacing):
        self.__vw_grid_spacing = vw_grid_spacing
        self._update_paths()

    def set_vw_grid_bounds_mode(self, vw_grid_bounds_mode):
        self.__vw_grid_bounds_mode = vw_grid_bounds_mode

    def set_vw_grid_offset(self, vw_grid_offset):
        self.__vw_grid_offset = vw_grid_offset

    def set_desc(self, desc):
        self.__desc = desc
        self._update_paths()

    def set_desc_cell_size(self, desc_cell_size):
        self.__desc_cell_size = desc_cell_size
        self._update_paths()

    def set_desc_cell_struct(self, desc_cell_struct):
        self.__desc_cell_struct = desc_cell_struct
        self._update_paths()

    def set_desc_cell_bins(self, desc_cell_bins):
        self.__desc_cell_bins = desc_cell_bins
        self._update_paths()

    def set_desc_smooth_sigma(self, desc_smooth_sigma):
        self.__desc_smooth_sigma = desc_smooth_sigma
        self._update_paths()

    def set_patch_page_thresh(self, patch_page_thresh):
        self.__patch_page_thresh = patch_page_thresh
        self._update_paths()

    def set_model_topology(self, model_topology):
        self.__model_topology = model_topology
        self._update_paths()

    def set_model_frame_state_fun(self, model_frame_state_fun):
        self.__model_frame_state_fun = model_frame_state_fun
        self._update_paths()

    def set_model_state_index(self, model_state_index):
        self.__model_state_index = model_state_index
        self._update_paths()

    def set_model_patch_size_reduction(self, model_patch_size_reduction):
        self.__model_patch_size_reduction = model_patch_size_reduction
        self._update_paths()

    def set_om_type(self, om_type):
        self.__om_type = om_type
        self._update_paths()

    def set_om_size(self, om_size):
        self.__om_size = om_size
        self._update_paths()

    def set_om_lineheights(self, om_lineheights):
        self.__om_lineheights = om_lineheights
        self._update_paths()

    def set_om_termination(self, om_termination):
        self.__om_termination = om_termination
        self._update_paths()

    def set_om_detann(self, om_detann):
        self.__om_detann = om_detann
        self._update_paths()

    def set_om_beamwidth(self, om_beamwidth):
        self.__om_beamwidth = om_beamwidth

    def set_om_smoothing(self, om_smoothing):
        self.__om_smoothing = om_smoothing
        self.__check_smooting_params(om_smoothing[0])
        self._update_paths()

    def set_sp_partitions(self, sp_partitions):
        self.__sp_partitions = sp_partitions
        self._update_paths()

    def set_sp_distance(self, sp_distance):
        self.__sp_distance = sp_distance
        self._update_paths()

    def set_ifs_mode(self, ifs_mode):
        self.__ifs_mode = ifs_mode
        self._update_paths()

    def set_ifs_frame_caching(self, use_frame_caching):
        self.__ifs_frame_caching = use_frame_caching

    def set_ifs_ght_cells(self, ifs_ght_cells):
        self.__ifs_ght_cells = ifs_ght_cells
        self._update_paths()

    def set_vocabulary_size(self, vocabulary_size):
        self.__vocabulary_size = vocabulary_size
        self._update_paths()

    def set_vocabulary_sampling_rate(self, vocabulary_sampling_rate):
        self.__vocabulary_sampling_rate = vocabulary_sampling_rate
        self._update_paths()

    def set_vocabulary_desc_contrast_thresh(self, vocabulary_desc_contrast_thresh):
        self.__vocabulary_desc_contrast_thresh = vocabulary_desc_contrast_thresh
        self._update_paths()

    def set_vocabulary_mode(self, vocabulary_mode):
        self.__vocabulary_mode = vocabulary_mode
        self._update_paths()

    def set_store_score_mat(self, store_score_mat):
        self.__store_score_mat = store_score_mat

    def set_store_retrieval_mat(self, store_retrieval_mat):
        self.__store_retrieval_mat = store_retrieval_mat

    def set_single_query_decoding(self, single_query_decoding):
        self.__single_query_decoding = single_query_decoding

    data_base = property(get_data_base, None, None, None)
    decoding_mode = property(get_decoding_mode, set_decoding_mode, None, None)
    vt_beamwidth = property(get_vt_beamwidth, set_vt_beamwidth, None, None)
    vt_problow = property(get_vt_problow, set_vt_problow, None, None)
    patch_page_thresh = property(get_patch_page_thresh, set_patch_page_thresh, None, None)
    model_topology = property(get_model_topology, set_model_topology, None, None)
    model_frame_state_fun = property(get_model_frame_state_fun, set_model_frame_state_fun, None, None)
    model_state_index = property(get_model_state_index, set_model_state_index, None, None)
    model_patch_size_reduction = property(get_model_patch_size_reduction, set_model_patch_size_reduction, None, None)
    om_type = property(get_om_type, set_om_type, None, None)
    om_size = property(get_om_size, set_om_size, None, None)
    om_lineheights = property(get_om_lineheights, set_om_lineheights, None, None)
    om_termination = property(get_om_termination, set_om_termination, None, None)
    om_detann = property(get_om_detann, set_om_detann, None, None)
    om_beamwidth = property(get_om_beamwidth, set_om_beamwidth, None, None)
    om_smoothing = property(get_om_smooting, set_om_smoothing, None, None)
    sp_partitions = property(get_sp_partitions, set_sp_partitions, None, None)
    sp_distance = property(get_sp_distance, set_sp_distance, None, None)
    ifs_mode = property(get_ifs_mode, set_ifs_mode, None, None)
    ifs_ght_cells = property(get_ifs_ght_cells, set_ifs_ght_cells, None, None)
    vocabulary_size = property(get_vocabulary_size, set_vocabulary_size, None, None)
    vocabulary_sampling_rate = property(get_vocabulary_sampling_rate, set_vocabulary_sampling_rate, None, None)
    vocabulary_desc_contrast_thresh = property(get_vocabulary_desc_contrast_thresh, set_vocabulary_desc_contrast_thresh, None, None)
    vocabulary_mode = property(get_vocabulary_mode, set_vocabulary_mode, None, None)
    patch_shift_denom = property(get_patch_shift_denom, set_patch_shift_denom, None, None)
    patch_overlap = property(get_patch_overlap, set_patch_overlap, None, None)
    patch_nms = property(get_patch_nms, set_patch_nms, None, None)
    patch_fitting = property(get_patch_fitting, set_patch_fitting, None, None)
    patch_ws_fitting = property(get_patch_ws_fitting, set_patch_ws_fitting, None, None)
    patch_hyp_region = property(get_patch_hyp_region, set_patch_hyp_region, None, None)
    patch_hyp_line = property(get_patch_hyp_line, set_patch_hyp_line, None, None)
    patch_hyp_config = property(get_patch_hyp_config, set_patch_hyp_config, None, None)
    term_weighting = property(get_term_weighting, set_term_weighting, None, None)
    frame_size = property(get_frame_size, set_frame_size, None, None)
    frame_step = property(get_frame_step, set_frame_step, None, None)
    frame_dir = property(get_frame_dir, set_frame_dir, None, None)
    vw_offset = property(get_vw_offset, set_vw_offset, None, None)
    vw_grid_spacing = property(get_vw_grid_spacing, set_vw_grid_spacing, None, None)
    vw_grid_bounds_mode = property(get_vw_grid_bounds_mode, set_vw_grid_bounds_mode, None, None)
    vw_grid_offset = property(get_vw_grid_offset, set_vw_grid_offset, None, None)
    desc = property(get_desc, set_desc, None, None)
    desc_cell_size = property(get_desc_cell_size, set_desc_cell_size, None, None)
    desc_cell_struct = property(get_desc_cell_struct, set_desc_cell_struct, None, None)
    desc_cell_bins = property(get_desc_cell_bins, set_desc_cell_bins, None, None)
    desc_smooth_sigma = property(get_desc_smooth_sigma, set_desc_smooth_sigma, None, None)
    store_score_mat = property(is_store_score_mat, set_store_score_mat, None, None)
    store_retrieval_mat = property(is_store_retrieval_mat, set_store_retrieval_mat, None, None)
    single_query_decoding = property(is_single_query_decoding, set_single_query_decoding, None, None)
    vw_accu_filter = property(get_vw_accu_filter, set_vw_accu_filter, None, None)
    gtp_encoding = property(get_gtp_encoding, set_gtp_encoding, None, None)

class QueryByStringConfig(WSConfig):


    def __init__(self, data_base_dir, tmp_base_dir, quantization_base_dir,
                 model_base_dir, ws_parameters_dict,
                 model_name, model_context_dep,
                 model_state_num, model_height_percentile, ws_estimation_mode):
        self.__model_name = model_name
        self.__model_context_dep = model_context_dep
        self.__model_state_num = model_state_num
        self.__model_height_percentile = model_height_percentile
        self.__ws_estimation_mode = ws_estimation_mode
        super(QueryByStringConfig, self).__init__(data_base_dir, tmp_base_dir,
                                                  quantization_base_dir,
                                                  model_base_dir,
                                                  **ws_parameters_dict)
        # Multiple height white space models are unsupported for query-by-string
        self.patch_ws_fitting = (0, 0, 1)


    def print_ws_config(self):

        ws_config_desc_list = [
        '\n',
        'model_name: %s' % self.__model_name,
        'model_context_dep: %s' % str(self.__model_context_dep),
        'model_state_num: %d' % self.__model_state_num,
        'model_height_percentile: %d' % self.__model_height_percentile,
        'ws_estimation_mode: %s' % self.__ws_estimation_mode,
        '\n']
        logger = logging.getLogger('QueryByStingConfig::INFO')
        logger.info('\n'.join(ws_config_desc_list))

        super(QueryByStringConfig, self).print_ws_config()

    def _generate_dirnames(self):
        dirname_tup = super(QueryByStringConfig, self)._generate_dirnames()
        dirname_list = list(dirname_tup)
        model_dep = 'cd' if self.__model_context_dep else 'ci'
        # model_topology and model_patch_size_reduction are defined in the
        # parent class as properties. Changes to their values will result in an
        # update of the paths.
        # --> _generate_dirnames in the inheriting class will be called.
        model_dirname_list = ['hmm-%s' % self.model_topology,
                              '-msn%d' % self.__model_state_num,
                              '-mhp%g' % self.__model_height_percentile,
                              '-%s' % model_dep,
                              '-psr%d-%d-%d-%d' %
                              self.model_patch_size_reduction,
                              '-ws_est-%s' % self.__ws_estimation_mode]
        dirname_list[3] = ''.join(model_dirname_list)
        return tuple(dirname_list)

    def is_model_context_dep(self):
        return self.__model_context_dep

    def get_model_name(self):
        return self.__model_name

    def get_model_state_num(self):
        return self.__model_state_num

    def get_model_height_percentile(self):
        return self.__model_height_percentile

    def get_ws_estimation_mode(self):
        return self.__ws_estimation_mode

    def has_ws_estimation_padding(self):
        return True if self.__ws_estimation_mode == 'pad' else False

    def has_ws_estimation_voting(self):
        return True if self.__ws_estimation_mode == 'vote' else False

    def has_ws_estimation_line(self):
        return True if self.__ws_estimation_mode == 'line' else False

    def set_model_context_dep(self, value):
        self.__model_context_dep = value
        super(QueryByStringConfig, self)._update_paths()

    def set_model_state_num(self, value):
        self.__model_state_num = value
        super(QueryByStringConfig, self)._update_paths()

    def set_model_height_percentile(self, value):
        self.__model_height_percentile = value
        super(QueryByStringConfig, self)._update_paths()

    def set_ws_estimation_mode(self, ws_estimation_mode):
        self.__ws_estimation_mode = ws_estimation_mode
        super(QueryByStringConfig, self)._update_paths()

    model_context_dep = property(is_model_context_dep, set_model_context_dep, None, None)
    model_state_num = property(get_model_state_num, set_model_state_num, None, None)
    model_height_percentile = property(get_model_height_percentile, set_model_height_percentile, None, None)
    model_name = property(get_model_name, None, None, None)
    ws_estimation_mode = property(get_ws_estimation_mode, set_ws_estimation_mode, None, None)

class ConfigArgParser(object):


    def __init__(self, published_properties=None):
        if published_properties is None:
            published_properties = self.get_published_properties()
        self.__published_properties = published_properties

    def publish_config(self, arg_parser, config):

        for prop in self.__published_properties:
            prop_key = '--%s' % prop
            prop_default = getattr(config, prop)
            help_msg = 'default: ' + str(prop_default)
            arg_parser.add_argument(prop_key, default=prop_default,
                                    type=self.__get_type(prop_default),
                                    help=help_msg)

    def set_arg_result(self, arg_result_ns, config):
        arg_result_dict = vars(arg_result_ns)
        for prop in self.__published_properties:
            setattr(config, prop, arg_result_dict[prop])
        return config

    @staticmethod
    def get_published_properties():
        return ['decoding_mode', 'vt_beamwidth', 'vt_problow',
                'model_state_index', 'model_topology',
                'om_type', 'om_size', 'om_lineheights', 'om_smoothing',
                'sp_distance',
                'ifs_mode', 'ifs_ght_cells',
                'model_frame_state_fun', 'model_patch_size_reduction',
                'vocabulary_size', 'vocabulary_sampling_rate',
                'vocabulary_desc_contrast_thresh', 'vocabulary_mode',
                'patch_shift_denom', 'patch_overlap', 'patch_nms',
                'patch_fitting', 'patch_hyp_region', 'patch_hyp_line',
                'term_weighting',
                'vw_accu_filter',
                'desc', 'desc_cell_size', 'desc_cell_struct', 'desc_cell_bins',
                'single_query_decoding']

    @staticmethod
    def __get_type(obj):
        #
        # *TODO*
        # resolve __get_type duplicate code (--> configio)
        # add parser for boolean value
        # ATTENTION: False values can only be specified as empty values currently
        #            see definition of python bool function.
        #            "if x is false or omitted this returns false"
        #
        obj_type = type(obj)
        if obj_type == tuple:
            return TupleIO.get_string_to_tuple_parser(obj)
        return obj_type


class TupleIO(object):

    @staticmethod
    def tuple_to_string(tup):
        return ','.join(str(e) for e in tup)

    @staticmethod
    def string_to_tuple(string, style):
        s2t = TupleIO.get_string_to_tuple_parser(style)
        return s2t(string)

    @staticmethod
    def get_string_to_tuple_parser(style):
        tup_type_list = [type(tup_elem) for tup_elem in style]
        return lambda tup_str: TupleIO.__parse_tuple(tup_str, tup_type_list)

    @staticmethod
    def __parse_tuple(tup_str, tup_type_list):
        split_tup_str = tup_str.split(',')
        if len(split_tup_str) != len(tup_type_list):
            raise ValueError("Tuple-String '%s' should have %d elements (has %d)" % (tup_str, len(tup_type_list), len(split_tup_str)))
        try:
            tup = tuple([type_fun(tup_elem) for type_fun, tup_elem in
                         zip(tup_type_list , split_tup_str)])
            return tup
        except ValueError:
            type_name_list = [tup_type.__name__ for tup_type in tup_type_list]
            raise TypeError("Tuple must be given as: '%s', but string is: '%s'" % (', '.join(type_name_list), tup_str))


class ConfigIO(object):

    @staticmethod
    def write_config_file(filepath, dictionary):
        ls = ['%s=%s' % (key, ConfigIO.__to_string(value)) for key, value in sorted(dictionary.items())]
        LineListIO.write_list(filepath, ls)

    @staticmethod
    def __to_string(obj):
        if type(obj) == tuple:
            return TupleIO.tuple_to_string(obj)
        else:
            return str(obj)

    @staticmethod
    def read_config_file(filepath, style_dict):
        ls = LineListIO.read_list(filepath)
        dictionary = {}
        for e in ls:
            i = e.find('=')
            if i == -1:
                raise ValueError("Missing '=' in '%s'" % e)
            key = e[:i]
            value = e[i + 1:]
            t = ConfigIO.__get_type(style_dict[key])
            dictionary[key] = t(value)
        return dictionary

    @staticmethod
    def __get_type(obj):
        obj_type = type(obj)
        if obj_type == tuple:
            return TupleIO.get_string_to_tuple_parser(obj)
        return obj_type
