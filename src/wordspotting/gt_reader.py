'''
Created on Mar 11, 2013

@author: lrothack
'''

import logging
import numpy as np
from patrec.serialization.list_io import LineListIO

class GroundTruthReader(object):
    '''
    Reader the gtp Ground Truth data format
    line-wise bounding box annotations.
    ul: upper left corner, lr: lower right corner

    ul_x ul_y lr_x lr_y annotation
    ...

    '''

    def __init__(self, base_path, gt_dir='GT/', gt_file_suffix='.gtp', ignore_missing_file=True,
                 gtp_encoding='ascii'):
        '''
        Constructor
        @param base_path: Base path of the experiment data (ending with /)
        @param gt_dir: Suffix of the ground truth files
        @param gt_file_suffix: Suffix of the ground truth files
        '''

        self.__base_path = base_path + gt_dir
        self.__gtp_encoding = gtp_encoding
        self.__gt_file_suffix = gt_file_suffix
        self.__ignore_missing_file = ignore_missing_file

    def get_base_path(self):
        return self.__base_path

    def get_document_gtp_filepath(self, document_name):
        return self.__base_path + document_name + self.__gt_file_suffix

    def read_ground_truth_document_bounds(self, document_name):
        '''
        Read ground truth and determine its enclosing/bounding rectangle in the
        document
        @param document_name: The name / identifier of the respective document
        @return: A tuple (x_min,y_min,x_max,y_max) specifying the bounding box
        '''
        gt_list = self.read_ground_truth(document_name)
        gt_bounds = [list(gt[1][0] + gt[1][1]) for gt in gt_list]
        gt_bounds_mat = np.array(gt_bounds)
        gt_doc_bounds = tuple(np.min(gt_bounds_mat[:, :2], axis=0)) + tuple(np.max(gt_bounds_mat[:, 2:], axis=0))
        return gt_doc_bounds

    def read_ground_truth(self, document_name):
        '''
        Read the ground truth information for the specified document. The file path
        is constructed based on the information specified in __init__
        @param document_name: The name of the respective document
        @return: A list of tuples
            ('word', ( (x_upperleft, y_upperleft), (x_lowerright, y_lowerright) ) )
        '''
        document_gt_path = self.get_document_gtp_filepath(document_name)
        gt_list = []
        listio = LineListIO()
        try:
            gt_list = listio.read_list(document_gt_path, encoding=self.__gtp_encoding)
        except ValueError as value_err:
            if self.__ignore_missing_file:
                gt_list = []
            else:
                raise value_err

        tuple_list = []
        for item in gt_list:
            item_list = item.split(None, 4)
            # cast numeric items to int
            item_list[0:4] = [int(x) for x in item_list[0:4]]
            item_tuple = (item_list[4], ((item_list[0], item_list[1]), (item_list[2], item_list[3])))
            tuple_list.append(item_tuple)

        return tuple_list

    def convert_ground_truth(self, document_name, filter_fun,
                           dst_suffix=''):
        logger = logging.getLogger('GroundTruthReader::convert_ground_truth')
        listio = LineListIO()
        # Append suffix in order to distinguish from original
        document_gt_dst_path = self.get_document_gtp_filepath(document_name)
        document_gt_dst_path += dst_suffix
        gt_src_tup_list = self.read_ground_truth(document_name)
        gt_dst_list = []
        for item in gt_src_tup_list:
            dst_item = filter_fun(item)
            transc = dst_item[0]
            if transc == '':
                logger.warn('Annotation "%s" empty after filtering..skipping',
                            item[0])
                continue
            bounds = dst_item[1]
            ul_x = bounds[0][0]
            ul_y = bounds[0][1]
            lr_x = bounds[1][0]
            lr_y = bounds[1][1]
            item_dst_str = '%d %d %d %d %s' % (ul_x, ul_y, lr_x, lr_y, transc)
            gt_dst_list.append(item_dst_str)

        listio.write_list(document_gt_dst_path, gt_dst_list,
                          encoding=self.__gtp_encoding)
