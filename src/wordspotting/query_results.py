"""
Created on Jan 18, 2013

@author: lrothack
"""
import logging
import os
import tqdm
import copy
import numpy as np
import cPickle as pickle
from patrec.serialization.matrix_io import MatrixIO
from patrec.serialization.list_io import LineListIO
from patrec.evaluation.retrieval import MeanAveragePrecision, MeanRecall, \
    IterativeMean, MeanInterpolatedPrecision
from patrec.evaluation.rectangle_intersection import RectangleIntersection
from bofhwr.wordspotting.definitions import ModelDefinitions
from bofhwr.wordspotting.gt_reader import GroundTruthReader


class RetrievalDataProvider(object):

    def __init__(self, config, buffered=True):
        logger = logging.getLogger('RetrievalDataProvider::__init__')
        logger.info('buffered=%s', str(buffered))
        self.__config = config
        self.__ret_arr_buffer = {}
        self.__buffered = buffered

    def retrieval_arr(self, document_id, query_id):
        logger = logging.getLogger('RetrievalDataProvider::retrieval_arr')
        ret_arr_fp = self.__config.get_retrieval_mat_filepath(document_id,
                                                              query_id)
        try:
            ret_arr = self.__ret_arr_buffer[ret_arr_fp]
        except KeyError:
            logger.debug('loading retrieval arr %s :: %s',
                         document_id, query_id)
            ret_arr = MatrixIO.read_matrix_as_byte(ret_arr_fp,
                                                   dim=6,
                                                   missing_dim=True,
                                                   missing_inner_dim=True)
            if self.__buffered:
                self.__ret_arr_buffer[ret_arr_fp] = ret_arr

        return ret_arr

    def retrieval_time(self, document_id, eval_id=None):
        retrieval_time_fp = self.__config.get_retrieval_time_filepath(document_id,
                                                                      eval_id)
        retrieval_time_list = LineListIO.read_list(retrieval_time_fp)
        # read mean retrieval time and mean per model retrieval time
        r_res_tup = tuple(float(r_time) for r_time in retrieval_time_list)
        r_patches, r_pm_patches, r_time, pm_r_time = r_res_tup
        return r_patches, r_pm_patches, r_time, pm_r_time


class RetrievalDataOverlapProvider(object):

    def __init__(self, ret_data_provider, dst_config,
                 retrieval_list, query_list=None):
        self.__ret_data_provider = ret_data_provider
        self.__patch_overlap = dst_config.patch_overlap

        gt_reader = GroundTruthReader(base_path=dst_config.data_base,
                                      gtp_encoding=dst_config.gtp_encoding)
        model_def = ModelDefinitions(gt_reader)

        gt_model_dict = model_def.generate_document_model_definitions(retrieval_list)
        if query_list is not None:
            query_model_dict = model_def.generate_document_model_definitions(query_list)
        else:
            query_model_dict = gt_model_dict

        self.__gt_model_dict = gt_model_dict
        self.__query_model_dict = query_model_dict

    def retrieval_time(self, document_id, eval_id=None):
        return self.__ret_data_provider.retrieval_time(document_id, eval_id)
    def retrieval_arr(self, document_id, query_id):
        retrieval_arr = self.__ret_data_provider.retrieval_arr(document_id,
                                                               query_id)
        query_name = self.__query_model_dict[query_id][1]
        ret_arr = self.process_retrieval_arr_overlap(self.__gt_model_dict,
                                                     retrieval_arr,
                                                     document_id,
                                                     query_name,
                                                     self.__patch_overlap)
        return ret_arr


    @staticmethod
    def process_retrieval_arr_overlap(gt_model_dict, retrieval_arr,
                                      retrieval_document, query_name,
                                      patch_overlap):
        gt_doc_model_query_bounds = [x[2] for x in gt_model_dict.values()
                                     if x[0] == retrieval_document and
                                     x[1] == query_name]
        rect_intersect = RectangleIntersection()

        # copy retrieval_arr in order to store updated relevance bits
        retrieval_arr = np.array(retrieval_arr)

        for retrieval_patch_item in retrieval_arr.transpose():
            # Get intersections with respective ground truth patches
            ul_x = retrieval_patch_item[2]
            ul_y = retrieval_patch_item[3]
            lr_x = retrieval_patch_item[4]
            lr_y = retrieval_patch_item[5]
            patch_xy_ul = (ul_x, ul_y)
            patch_xy_lr = (lr_x, lr_y)
            patch_rel = rect_intersect.intersects((patch_xy_ul, patch_xy_lr),
                                                  gt_doc_model_query_bounds,
                                                  patch_overlap)
            retrieval_patch_item[1] = patch_rel

        return retrieval_arr


class QueryResultsProcessor(object):

    def __init__(self, config, retrieval_list, query_list, buffered):
        self.__config = config
        self.__data_provider = RetrievalDataProvider(config, buffered)
        self.__retrieval_list = retrieval_list
        self.__query_list = query_list


    def evaluate(self, patch_overlap, eval_id=None, store_results=True):
        logger = logging.getLogger('QueryResultsProcessor::evaluate')
        logger.info('')
        logger.info('patch_overlap= %g, eval_id=%s',
                    patch_overlap,
                    str(eval_id))
        if self.__config.patch_overlap != patch_overlap:
            config_dst = copy.deepcopy(self.__config)
            config_dst.patch_overlap = patch_overlap
            logger.info('\nTarget config:')
            config_dst.print_ws_config_id()
            scores_path = config_dst.get_scores_path()
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)

            data_provider = RetrievalDataOverlapProvider(self.__data_provider,
                                                         config_dst,
                                                         self.__retrieval_list,
                                                         self.__query_list)
        else:
            config_dst = self.__config
            data_provider = self.__data_provider


        ws_eval = QueryResultsEvaluator(config_dst,
                                        self.__retrieval_list,
                                        eval_id,
                                        data_provider)
        q_res_tup = ws_eval.evaluate_querylist(self.__query_list, store_results)
        t_res_tup = ws_eval.evaluate_retrieval_time(store_results)

        return q_res_tup, t_res_tup

    def export_results(self, eval_id=None):
        logger = logging.getLogger('QueryResultsProcessor::export_results')
        ws_eval = QueryResultsEvaluator(self.__config,
                                        self.__retrieval_list,
                                        eval_id,
                                        self.__data_provider)
        result_dict = ws_eval.query_retrieval_mat_dict(self.__query_list)
        score_thresh_factor = 1.0
        export_fp = self.__config.get_query_results_export_filepath(score_thresh_factor,
                                                                    eval_id)
        logger.info('writing to %s', export_fp)
        with open(export_fp, 'wb') as file_handle:
            pickle.dump(result_dict, file_handle, protocol=2)


    def delete_partial_results(self, eval_id=None):
        logger = logging.getLogger('QueryResultsProcessor::delete_partial_results')
        ws_eval = QueryResultsEvaluator(self.__config,
                                        self.__retrieval_list,
                                        eval_id,
                                        self.__data_provider)
        logger.info('processing partial query and timing results...')
        ws_eval.delete_partial_results(self.__query_list)

class QueryResultsEvaluator(object):
    """ Evaluates retrieval matrices.

        --> calculates mean average precision, mean recall and their statistics.
    """

    def __init__(self, config, retrieval_document_list, eval_id=None,
                 ret_data_provider=None):
        """

        Params:
            config: WSConfig Object or Config Object equal interface.
            retrieval_document_list: List of documents the retrieval was
                performed on e.g. [2700270, ... ] for gw.
            eval_id: If evaluation is splitted to multiple parts provide
                the part id.
        """

        if retrieval_document_list is None:
            raise ValueError('document lists (query or retrieval) ' +
                             'must not be None!')
        if ret_data_provider is None:
            ret_data_provider = RetrievalDataProvider(config, buffered=False)

        self.__ret_data_p = ret_data_provider

        self.__config = config
        self.__retrieval_document_list = retrieval_document_list
        self.__eval_id = eval_id


        gt_reader = GroundTruthReader(base_path=self.__config.data_base,
                                      gtp_encoding=self.__config.gtp_encoding)
        self.__model_definitions = ModelDefinitions(gt_reader)

    def evaluate_retrieval_time(self, store_results=True):
        logger = logging.getLogger(
            'QueryResultsEvaluator::evaluate_retrieval_time')
        mean_retrieval_patches = IterativeMean()
        mean_permodel_retrieval_patches = IterativeMean()
        mean_retrieval_time = IterativeMean()
        mean_permodel_retrieval_time = IterativeMean()
        for document_id in self.__retrieval_document_list:
            ret_times = self.__ret_data_p.retrieval_time(document_id,
                                                         self.__eval_id)
            r_patches, pm_r_patches, r_time, pm_r_time = ret_times
            mean_retrieval_patches.add_value(r_patches)
            mean_permodel_retrieval_patches.add_value(pm_r_patches)
            mean_retrieval_time.add_value(r_time)
            mean_permodel_retrieval_time.add_value(pm_r_time)
        time_result_list = []
        if self.__config.single_query_decoding:
            time_result_list.append(
                'INDIVIDUAL QUERY DECODING TIME MEASUREMENT')
        else:
            time_result_list.append('MULTI QUERY DECODING TIME MEASUREMENT')
            time_result_list.append(' (per model retrieval time does NOT '
                                    'include patch processing per model)')
        m_ret_patches = mean_retrieval_patches.get_mean()
        m_pm_ret_patches = mean_permodel_retrieval_patches.get_mean()
        m_ret_time = mean_retrieval_time.get_mean()
        m_pm_ret_time = mean_permodel_retrieval_time.get_mean()
        time_result_list.append('MEAN RETRIEVAL PATCHES: %g' % m_ret_patches)
        time_result_list.append('MEAN MODEL RETRIEVAL PATCHES: %g' %
                                m_pm_ret_patches)
        time_result_list.append('MEAN RETRIEVAL TIME: %.6f' % m_ret_time)
        time_result_list.append('MEAN MODEL RETRIEVAL TIME: %.6f' %
                                m_pm_ret_time)

        if store_results:
            listio = LineListIO()
            # Store results list
            time_results_list_filepath = self.__config.get_time_results_filepath(self.__eval_id)
            logger.info('Going to write time results list: %s',
                        time_results_list_filepath)
            listio.write_list(time_results_list_filepath, time_result_list)

        logger.info('\n'.join(time_result_list) + '\n')
        return m_ret_patches, m_pm_ret_patches, m_ret_time, m_pm_ret_time

    def __get_model_definitions(self, document_list):
        model_dict = self.__model_definitions.generate_document_model_definitions(document_list)
        return model_dict

    def evaluate_retrievallist(self, store_results=True):
        return self.evaluate_querylist(self.__retrieval_document_list,
                                       store_results)

    def evaluate_querylist(self, query_list, store_results=True):
        query_model_dict = self.__get_model_definitions(query_list)
        gt_model_dict = self.__get_model_definitions(
            self.__retrieval_document_list)

        return self.evaluate(query_model_dict, gt_model_dict, store_results)

    def query_retrieval_mat_dict(self, query_list):
        """

        :param query_list: Document ids of documents the queries come from.
        :return:
        """
        query_model_dict = self.__get_model_definitions(query_list)

        query_retrieval_dict = {}
        for query_id, query_def in sorted(query_model_dict.items()):

            # Load retrieval results for the current query and all documents
            retrieval_mat_list = []
            retrieval_doc_list = []
            for document_id in self.__retrieval_document_list:
                ret_mat_part = self.__ret_data_p.retrieval_arr(document_id,
                                                               query_id)
                retrieval_mat_list.append(ret_mat_part)
                document_exp_list = [document_id] * ret_mat_part.shape[1]
                retrieval_doc_list.append(document_exp_list)

            retrieval_document_mat = np.hstack(tuple(retrieval_doc_list))
            retrieval_mat = np.hstack(tuple(retrieval_mat_list))
            # Sort it according to scores
            retrieval_mat_scores = retrieval_mat[0, :]
            retrieval_mat_scores_sort_ind = np.argsort(retrieval_mat_scores)
            retrieval_mat_sort = retrieval_mat[:, retrieval_mat_scores_sort_ind]
            ret_document_mat = retrieval_document_mat[retrieval_mat_scores_sort_ind]

            ret_patch_mat = retrieval_mat_sort

            query_retrieval_dict[query_id] = (query_def,
                                              ret_patch_mat,
                                              ret_document_mat)

        return query_retrieval_dict

    def evaluate(self, query_model_dict, gt_model_dict, store_results=True):
        '''
        '''
        logger = logging.getLogger('QueryResultsEvaluator::evaluate')
        logger.info('initializing..')
        score_thresh_factor = 1.0
        config = self.__config
        list_m_ap = MeanAveragePrecision()
        m_ap = MeanAveragePrecision(max_interp=False)
        m_iap = MeanAveragePrecision(max_interp=True)
        m_ip = MeanInterpolatedPrecision()
        m_r = MeanRecall()
        m_od = IterativeMean()

        query_result_list = []
        query_result_dict = {}

        for query_id, query_def in tqdm.tqdm(query_model_dict.items()):
            gt_query_model_num = len([x for x in gt_model_dict.values()
                                      if x[1] == query_def[1]])

            # Load retrieval results for the current query and all documents
            m_od_doc = IterativeMean()
            retrieval_mat_list = []
            for document_id in self.__retrieval_document_list:
                # Read retrieval matrix for the <query_id>
                # that has been evaluated on <document_id>
                #
                # Retrieval results have the form: 6xN ndarray for N patches
                # the six rows correspond to:
                # 0: retrieval score, smaller values are better (indicate higher
                #     similarity to the query) than larger values.
                # 1: relevance bit, 1 indicates that the patch is relevant to
                #    the query, 0 indicates that the patch is irrelevant.
                # 2-5: patch coordinates in pixels, ul_x, ul_y, lr_x, lr_y
                #      x corresponds to image columns,
                #      y corresponds to image rows.
                #      ul: upper left, lr: lower right patch corner
                #
                # ATTENTION: if no results could be obtained, ret_mat_part
                # will be empty. It is important that the shape of the empty
                # array is (6, 0), i.e., array([], shape=(6, 0), dtype=float32)
                # compare MatrixIO.read_matrix_as_byte
                ret_mat_part = self.__ret_data_p.retrieval_arr(document_id,
                                                               query_id)

                # Check for over-detection and register value for calculating
                # the mean over all documents for this query
                #
                # IMPORTANT: over-detected items will be marked as irrelevant!
                od_score = self.check_overdetection(gt_model_dict,
                                                    document_id,
                                                    query_def,
                                                    ret_mat_part)
                m_od_doc.add_value(od_score)
                # Append corrected (over-detection) retrieval_mat to list
                retrieval_mat_list.append(ret_mat_part)

            # Update the mean over-detection score
            m_od.add_value(m_od_doc.get_mean())
            # Build the retrieval matrix for the query over all documents
            # Empty arrays can be stacked as long as their shape matches,
            # see above
            retrieval_mat = np.hstack(tuple(retrieval_mat_list))
            # Sort it according to scores
            retrieval_mat_scores = retrieval_mat[0, :]
            retrieval_mat_relevance = retrieval_mat[1, :]
            retrieval_mat_scores_sort_ind = np.argsort(retrieval_mat_scores)
            retrieval_mat_scores_sort = retrieval_mat_scores[retrieval_mat_scores_sort_ind]

            if score_thresh_factor < 1.0:
                # Determine threshold according to score_thresh_factor
                retrieval_mat_scores_min = retrieval_mat_scores_sort[0]
                retrieval_mat_scores_max = retrieval_mat_scores_sort[retrieval_mat_scores_sort.size - 1]
                retrieval_mat_scores_diff = (retrieval_mat_scores_max -
                                             retrieval_mat_scores_min)
                retrieval_mat_scores_thresh = (retrieval_mat_scores_min +
                                               (score_thresh_factor *
                                                retrieval_mat_scores_diff))
                thresh_index = np.searchsorted(retrieval_mat_scores_sort,
                                               retrieval_mat_scores_thresh)
            else:
                thresh_index = retrieval_mat_scores_sort_ind.size

            # Perform thresholding
            retrieval_mat_scores_sort_ind = retrieval_mat_scores_sort_ind[:thresh_index]
            retrieval_mat_relevance_sort = retrieval_mat_relevance[retrieval_mat_scores_sort_ind]
            # Calculate retrieval scores (average precision, recall)
            query_list_ap = list_m_ap.average_precision(retrieval_mat_relevance_sort,
                                                        gt_relevance_num=None)
            query_ap = m_ap.average_precision(retrieval_mat_relevance_sort,
                                              gt_query_model_num)
            query_iap = m_iap.average_precision(retrieval_mat_relevance_sort,
                                                gt_query_model_num)
            m_ip.interpolated_precision(retrieval_mat_relevance_sort,
                                        gt_query_model_num)
            query_r = m_r.recall(retrieval_mat_relevance_sort,
                                 gt_query_model_num)
            # Append query and scores to results list
            query_result = '%s %.6f %.6f %.6f %.6f' % (query_id,
                                                       query_list_ap,
                                                       query_r,
                                                       query_ap,
                                                       query_iap)
            query_result_list.append(query_result)
            query_result_dict[query_id] = (query_r,
                                           query_ap,
                                           query_iap)


        eval_result_list = []
        # If more than a single query has been decoded: perform length specific
        # evaluation
        if len(query_result_dict) > 1:
            len_eval = QueryLengthResults(query_model_dict)
            res_tup = len_eval.len_eval(query_result_dict)
            score_ids = ['mR', 'mAP', 'mIAP']
            eval_len_list = len_eval.eval_desc_list(*res_tup,
                                                    score_ids=score_ids)
            eval_len_list.append('\n')
            eval_result_list.extend(eval_len_list)

        eval_result_list.append('Mean Average Precision: %.4f' %
                                m_ap.get_mean())
        eval_result_list.append('Mean Recall: \t %.4f' %
                                m_r.get_mean())
        eval_result_list.append('Mean Over-detection: \t %.4f' %
                                m_od.get_mean())
        eval_result_list.append('Mean Interpolated Average Precision: %.4f' %
                                m_iap.get_mean())
        eval_result_list.append('Mean Interpolated Precision:')
        m_ip_mean = m_ip.get_mean()
        eval_result_list.append('[ %s ]' % ', '.join(['%.2f' % p_r
                                                      for p_r in m_ip_mean]))
        eval_result_list.append('( List Mean Average Precision: %.4f )' %
                                list_m_ap.get_mean())

        if store_results:
            listio = LineListIO()
            # Store results list
            query_results_list_filepath = config.get_query_results_list_filepath(score_thresh_factor,
                                                                                 self.__eval_id)
            logger.info('Going to write query results list: %s',
                        query_results_list_filepath)
            listio.write_list(query_results_list_filepath, query_result_list)

            eval_results_filepath = config.get_eval_results_filepath(score_thresh_factor,
                                                                     self.__eval_id)
            logger.info('Going to write evaluation results: %s',
                        eval_results_filepath)
            listio.write_list(eval_results_filepath, eval_result_list)

        eval_out = ['\n###FINISHED###',
                    '#######################################################\n',
                    '\n'.join(eval_result_list),
                    '\n']
        logger.info(''.join(eval_out))
        return (list_m_ap.get_mean(),
                m_r.get_mean(),
                m_od.get_mean(),
                m_ap.get_mean(),
                m_iap.get_mean(),
                m_ip_mean)

    def check_overdetection(self, gt_model_dict, document_id,
                            query_def, retrieval_mat):
        '''
        @requires: Entries in retrieval_mat must be sorted according to their
            score values <best scores> --> <worst scores>
        @postcondition: retrival_mat is modified: elements detecting an already
            detected ground truth item are marked as irrelevant.
        @return: over-detection score
        '''
        logger = logging.getLogger(
            'QueryResultsEvaluator::check_overdetection')
        rect_intersect = RectangleIntersection()

        patch_overlap = self.__config.patch_overlap

        query_name = query_def[1]

        #
        # Filter ground truth for query_name
        #
        gt_doc_model_query_bounds = [x[2] for x in gt_model_dict.values()
                                     if x[0] == document_id and
                                     x[1] == query_name]

        # Allocate data structure for registering detections
        gt_query_accu = np.zeros(len(gt_doc_model_query_bounds))

        for retrieval_patch_item in retrieval_mat.transpose():
            # Check if retrieved patch is relevant
            if retrieval_patch_item[1] == 1.0:
                # Get intersections with respective ground truth patches
                ul_x = retrieval_patch_item[2]
                ul_y = retrieval_patch_item[3]
                lr_x = retrieval_patch_item[4]
                lr_y = retrieval_patch_item[5]
                patch_xy_ul = (ul_x, ul_y)
                patch_xy_lr = (lr_x, lr_y)
                intersect_list = rect_intersect.intersection((patch_xy_ul,
                                                              patch_xy_lr),
                                                             gt_doc_model_query_bounds,
                                                             patch_overlap)
                intersect_mask = np.array(intersect_list)

                # Check for error conditions
                intersect_num = intersect_mask.sum()
                if intersect_num == 0:
                    raise ValueError('FATAL: Detected false relevant item!')
                elif intersect_num > 1:
                    logger.warn('ATTENTION: patch intersects with more ' +
                                'than one item from GT!')

                gt_query_accu[intersect_mask] += 1
                # If GT item has already been detected:
                # make retrieved patch irrelevant
                if (gt_query_accu[intersect_mask] > 1).any():
                    retrieval_patch_item[1] = 0.0

        #
        # Determine over-detection rate as the fraction of over-detections and
        # all detections
        #
        # Total detection number
        #
        total_detections = gt_query_accu.sum()

        #
        # Determine over-detection number
        #

        # Remove regular detections
        gt_query_accu_od_mask = gt_query_accu > 0
        gt_query_accu[gt_query_accu_od_mask] -= 1

        # Over-detection number
        over_detections = gt_query_accu.sum()

        if total_detections > 0.0:
            over_detection_score = over_detections / float(total_detections)
        else:
            over_detection_score = 0.0

        return over_detection_score

    def delete_partial_results(self, query_list):
        self.delete_partial_timing_results()
        query_model_dict = self.__get_model_definitions(query_list)
        query_id_list = sorted(query_model_dict.keys())
        self.delete_partial_retieval_results(query_id_list)

    def delete_partial_retieval_results(self, query_id_list):
        logger = logging.getLogger(
            'QueryResultsEvaluator::delete_partial_retrieval_results')
        logger.info('deleting...')
        for query_id in tqdm.tqdm(query_id_list):
            for document_id in self.__retrieval_document_list:
                retrieval_mat_fp = self.__config.get_retrieval_mat_filepath(document_id,
                                                                            query_id)
                os.remove(retrieval_mat_fp)

    def delete_partial_timing_results(self):
        logger = logging.getLogger(
            'QueryResultsEvaluator::delete_partial_timing_results')
        logger.info('deleting...')
        for document_id in self.__retrieval_document_list:
            retrieval_time_fp = self.__config.get_retrieval_time_filepath(document_id,
                                                                          self.__eval_id)
            os.remove(retrieval_time_fp)


class QueryResultsReader(object):

    def __init__(self, config, query_document_list):
        self.__config = config

        data_base = config.data_base
        gt_reader = GroundTruthReader(base_path=data_base,
                                      gtp_encoding=config.gtp_encoding)
        model_definitions = ModelDefinitions(gt_reader)
        self.__query_model_dict = model_definitions.generate_document_model_definitions(
            query_document_list)

    def read_query_results(self, score_thresh_factor=1.0):
        logger = logging.getLogger('QueryResultsReader::read_query_results')
        listio = LineListIO()

        query_results_list_filepath = self.__config.get_query_results_list_filepath(
            score_thresh_factor)
        logger.info('  %s', query_results_list_filepath)
        query_results_list = listio.read_list(query_results_list_filepath)

        query_results_list_split = [item.split()
                                    for item in query_results_list]

        query_results_dict = {}
        for query_result in query_results_list_split:
            # Map scores to float
            query_result_scores = tuple(float(x) for x in query_result[1:])
            query_results_dict[query_result[0]] = query_result_scores

        return query_results_dict


class QueryResultsFilter(object):

    def __init__(self, config, document_list=None):
        '''
        '''
        if document_list == None:
            document_list = config.default_document_list()

        self.__results_reader = QueryResultsReader(config, document_list)

    def filter_by_len(self, model_len, score_thresh_factor):
        logger = logging.getLogger('QueryResultsFilter::filter_by_len')
        query_results_dict = self.__results_reader.read_query_results(
            score_thresh_factor)
        query_list = [x for x in query_results_dict.values()
                      if len(x[0][1]) == model_len]
        logger.info(query_list)

    def filter_by_model(self, model_name, score_thresh_factor):
        logger = logging.getLogger('QueryResultsFilter::filter_by_model')
        query_results_dict = self.__results_reader.read_query_results(
            score_thresh_factor)
        query_list = [x for x in query_results_dict.values()
                      if x[0][1] == model_name]
        logger.info(query_list)


class QueryLengthResults(object):

    def __init__(self, query_model_dict, model_id_stoplist=None):
        '''
        '''

        self.__query_model_dict = query_model_dict

        if model_id_stoplist is not None:
            self.__query_model_dict = ModelDefinitions.filter_model_dict(self.__query_model_dict,
                                                                         model_id_stoplist)

    @staticmethod
    def mAP_mR_eval(query_results_dict):
        ap_r_mapper = lambda x: (x[2], x[1])
        ap_r_list = [ap_r_mapper(x)
                     for x in query_results_dict.values()]
        ap_r_mat = np.array(ap_r_list)
        m_ap = np.mean(ap_r_mat[:, 0])
        m_r = np.mean(ap_r_mat[:, 1])
        logger = logging.getLogger('QueryResults::mAP_mR_eval')
        logger.info('Mean Average Precision: %f\t', m_ap)
        logger.info('Mean Recall: \t%f', m_r)
        return m_ap, m_r

    def len_eval(self, query_result_dict, len_mapper=None):
        if len_mapper is None:
            len_mapper = lambda x: len(self.__query_model_dict[x][1])

        q_id_list = sorted(query_result_dict.keys())
        # 2D column vector containing query length values
        q_len_arr = np.array([len_mapper(q_id) for q_id in q_id_list])
        # 2D matrix containing query scores in each row
        q_score_arr = np.array([list(query_result_dict[q_id])
                                for q_id in q_id_list])

        # Vector storing the different existing lengths
        len_unique = np.unique(q_len_arr)
        # Accumulate scores for queries of equal lengths
        scores_arr = np.zeros((len(len_unique), q_score_arr.shape[1]))
        # Store number of queries of individual length in addition
        len_count_arr = np.zeros(len(len_unique))
        for idx, q_len in enumerate(len_unique):
            q_len_mask = q_len_arr == q_len
            n_q_len = np.sum(q_len_mask)
            len_count_arr[idx] = n_q_len
            scores_arr[idx, :] = np.sum(q_score_arr[q_len_mask, :], axis=0)
            scores_arr[idx, :] /= n_q_len


        return len_unique, len_count_arr, scores_arr

    @staticmethod
    def eval_desc_list(len_unique, len_count_arr, scores_arr,
                       score_ids=None):
        eval_str_list = []
        eval_str_list.append('Query lengths')
        score_lst_str = '   '.join('%02d' % u_len for u_len in len_unique)
        eval_str_list.append('[ %s ]' % score_lst_str)
        eval_str_list.append('Query length counts')
        score_lst_str = '   '.join('%02d' % c_u_len for c_u_len in len_count_arr)
        eval_str_list.append('[ %s ]' % score_lst_str)

        if score_ids is not None:
            for idx, score_col in enumerate(scores_arr.T):
                eval_str_list.append('%s:' % score_ids[idx])
                score_lst_str = ' '.join('%.2f' % score for score in score_col)
                eval_str_list.append('[ %s ]' % score_lst_str)
        else:
            for idx, score_row in enumerate(scores_arr):
                eval_str_list.append('length: %d', len_unique[idx])
                score_lst_str = ' '.join('%.2f' % score for score in score_row)
                eval_str_list.append('[ %s ]' % score_lst_str)

        return eval_str_list
