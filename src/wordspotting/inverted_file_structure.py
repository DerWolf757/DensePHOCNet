'''
Created on Mar 10, 2014

@author: leonard
'''

import logging
import collections
import tqdm
import numpy as np
from bofhwr.wordspotting._ifs_accu import IFSAccu

class InvertedFrameStructure(object):

    __N_CACHE_INITIAL = 20

    def __init__(self, line_seq_gen, doc_patch_gen, patch_quant, frame_spec,
                 model_size, cache_frames=False):
        logger = logging.getLogger('InvertedFrameStructure::__init__')
        self.__height_ifslist_dict = {}

        f_size, f_step, f_dir = frame_spec

        self.__line_seq_gen = line_seq_gen
        self.__line_width = doc_patch_gen.line_width()
        xy_min, xy_max = line_seq_gen.get_vw_grid_bounds()

        x_pos_list = range(xy_min[0], xy_max[0] + 1, f_step)

        if f_dir == 'rl':
            x_pos_list = x_pos_list[::-1]

        self.__x_pos_list = x_pos_list
        self.__x_offset = int(f_size / 2.0)

        self.__doc_patch_gen = doc_patch_gen
        self.__patch_quant = patch_quant
        self.__model_size = model_size
        if cache_frames:
            logger.info('caching frames and index')
            heights_list = line_seq_gen.get_heights_list()
            heights_arr = np.array(heights_list)
            heights_arr = np.sort(heights_arr)
            heights_arr = heights_arr[:InvertedFrameStructure.__N_CACHE_INITIAL]
            for height in tqdm.tqdm(heights_arr):
                self.__height_ifslist_cache(height)
        else:
            logger.info('index will be built upon request')


    def __height_ifslist(self, height):
        """ATTENTION: function requires an extreme amount of memory and is
        inefficient. Data needs to be copied from C++ layer (twice)
        See svq export and ifs_list to ifs_arr_list conversion
        Provide a native C++ datastructure that can directly handle svq C++
        objects without copying and can directly accessed from C++
        See ifs_accu.hpp. Access from Python is not required necessarily.
        """
        ifs_list = [collections.deque() for _ in range(self.__model_size)]

        # line_width and height must be aligned with vw grid
        patch_size_snp = (self.__line_width, height)
        patch_shift_snp = self.__patch_quant.get_patch_shift_snp(patch_size_snp)
        p_seq_tup = self.__line_seq_gen.generate_patch_sequences(patch_size_snp,
                                                          patch_shift_snp,
                                                          height)
        _, y_pos_list, _ = p_seq_tup
        y_offset = int(height / 2.0)
        svq_list = self.__doc_patch_gen.get_line_representations(y_pos_list,
                                                                 height)
        for y_pos, svq in zip(y_pos_list, svq_list):
            svq_frame_list = []
            svq.export_vectors_list(svq_frame_list)
            for x_pos, frame in zip(self.__x_pos_list, svq_frame_list):
                for vw, score in frame:
                    ifs_list[vw].append([x_pos + self.__x_offset,
                                         y_pos + y_offset,
                                         score])

        # transform py xys_list to xys_arr
        ifs_arr_list = [np.array(xys_lst, dtype=np.float32)
                        for xys_lst in ifs_list]
        return ifs_arr_list

    def __height_ifslist_cache(self, height):
        if height in self.__height_ifslist_dict:
            height_ifslist = self.__height_ifslist_dict[height]
        else:
            height_ifslist = self.__height_ifslist(height)
            self.__height_ifslist_dict[height] = height_ifslist

        return height_ifslist

    def ifs_list(self, height):
        line_heights = self.__line_seq_gen.get_patch_heights((0, height))
        if len(line_heights) == 0:
            raise ValueError('No line hyps could be found for patch height %d',
                             height)
        elif len(line_heights) > 1:
            raise ValueError('multi line hyp processing is unsupported!')
        line_height = line_heights[0]

        return self.__height_ifslist_cache(line_height)

    def xys_arr(self, height, visualword):
        '''
        @return: an numpy array containing row-wise xy coordinates
        '''
        ifs_list = self.ifs_list(height)
        return ifs_list[visualword]


class InvertedFileStructure(object):
    '''
    classdocs
    '''


    def __init__(self, visualword_array, model_size):
        '''
        Constructor
        '''
        #
        # Build inverted file structure index
        #
        ifs_keys = set(visualword_array[:, 2])
        self.__ifs_list = []
        for vw in range(model_size):
            ifs_key_arr = np.array([], dtype=np.uint16)
            if vw in ifs_keys:
                vs_ind = visualword_array[:, 2] == vw
                ifs_key_arr = np.ones(3, dtype=np.float32)
                ifs_key_arr[:2] = visualword_array[vs_ind, :2]

            self.__ifs_list.append(ifs_key_arr)

    def ifs_list(self, _):
        return self.__ifs_list

    def xys_arr(self, _, visualword):
        '''
        @return: an numpy array containing row-wise xy coordinates
        '''
        return self.__ifs_list[visualword]


class InvertedFileStructureAccu(object):

    def __init__(self, ifs, patch_seq, accu_cell_size, query_size):
        self.__ifs = ifs
        #
        # Initialize accumulator data structure
        #
        self.__accu_cell_size = accu_cell_size
        self.__accu_yx_mat, accu_mat_dim = patch_seq.get_patch_matrix(accu_cell_size,
                                                                      accu_cell_size)
        self.__vw_grid_bounds = patch_seq.get_vw_grid_bounds()
        query_size_mat = np.array(list(query_size))
        accu_offset = np.array(np.around(query_size_mat[[1, 0]] * 0.5),
                               dtype=np.int64)
        self.__accu_yx_mat -= accu_offset
        self.__accu_mat_dim = accu_mat_dim
        self.__x_ref = accu_offset[1]
        self.__query_width = query_size_mat[0]
        self.__query_height = query_size_mat[1]
        self.__ifs_accu = IFSAccu(self.__vw_grid_bounds[0][0],
                                  self.__vw_grid_bounds[0][1],
                                  self.__accu_cell_size[0],
                                  self.__accu_cell_size[1],
                                  self.__query_width,
                                  self.__x_ref)

    def accu_yx_mat(self):
        return self.__accu_yx_mat

    def accu_mat_dim(self):
        return self.__accu_mat_dim

    def accumulate_ght(self, state_vw_arr):
        accu_mat = np.zeros(self.__accu_mat_dim, dtype='float32')
        ifs_arr_list = self.__ifs.ifs_list(self.__query_height)
        self.__ifs_accu.accumulate_ght(ifs_arr_list,
                                       state_vw_arr, accu_mat)
        return accu_mat

    def accumulate_ght_py(self, state_vw_arr):
        '''
        Lookup visual word locations in ifs and
        increment accus in corresponding cells
        '''
        if state_vw_arr.ndim != 2:
            raise ValueError('state_vw_arr must be a 2d ndarray')
        n_states = float(state_vw_arr.shape[0])
        vw_bounds_ul = self.__vw_grid_bounds[0]
        accu_mat = np.zeros(self.__accu_mat_dim, dtype='float32')
        ifs = self.__ifs
        accu_mat_dim = self.__accu_mat_dim
        accu_cell_size = self.__accu_cell_size
        query_width = self.__query_width
        query_height = self.__query_height
        state_area_cen = (query_width / n_states) / 2.0
        for state_index, vw_arr in enumerate(state_vw_arr):
            x_offset = int(self.__x_ref -
                           (((state_index / n_states) * query_width) +
                           state_area_cen))
            for vw, mass in enumerate(vw_arr):
                if mass == 0:
                    continue

                xys_arr = ifs.xys_arr(query_height, vw)
                for xys in xys_arr:
                    # Always floor coordinates: accu index refers to upper-left patch corner
                    c_accu = int((xys[0] + x_offset - vw_bounds_ul[0]) / accu_cell_size[0])
                    r_accu = int((xys[1] - vw_bounds_ul[1]) / accu_cell_size[1])
                    if (r_accu >= 0 and c_accu >= 0 and
                        r_accu < accu_mat_dim[0] and c_accu < accu_mat_dim[1]):
                        accu_mat[r_accu, c_accu] += mass * xys[2]

        return accu_mat

