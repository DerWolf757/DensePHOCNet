'''
Created on Jan 27, 2013

@author: leonard
'''
import os
import numpy as np
import scipy.misc
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib
from bofhwr.wordspotting.query_results import QueryResultsEvaluator
from patrec.caching.decorator_cache import FifoCache
from bofhwr.wordspotting.gt_reader import GroundTruthReader
from patrec.features.normalization import normalize_image

class QueryResultsVisualization(object):

    def __init__(self, config, retrieval_document_list, export_dir):
        self.__config = config
        self.__query_eval = QueryResultsEvaluator(config, retrieval_document_list)
        self.__export_dir = export_dir

    def visualize_query_results(self, query_id_list, query_document_list, n_top_results=10):

        query_result_dict = self.__query_eval.query_retrieval_mat_dict(query_document_list)
        for query_id in query_id_list:
            query_path = self.__export_dir + query_id + '/'

            if not os.path.exists(query_path):
                os.makedirs(query_path)
            q_def, ret_patch_mat, ret_doc_mat = query_result_dict[query_id]
            q_document, _, q_bounds = q_def
            q_fn_base = '%s' % query_id
            filename_basepath = query_path + q_fn_base
            self.__export_bounds(q_document, q_bounds, filename_basepath)

            for index in range(n_top_results):
                p_document = ret_doc_mat[index]

                p_relevance = ret_patch_mat[1, index]

                p_rel_id = 'nr' if p_relevance == 0 else 'r'
                p_fn_base = '%s_p_%02d_%s' % (query_id, index, p_rel_id)

                p_bounds_ul = tuple(ret_patch_mat[2:4, index])
                p_bounds_lr = tuple(ret_patch_mat[4:6, index])
                p_bounds = (p_bounds_ul, p_bounds_lr)
                filename_basepath = query_path + p_fn_base
                self.__export_bounds(p_document, p_bounds, filename_basepath)

    @FifoCache(10)
    def __document_image_mat(self, document_name):
        page_img_path = self.__config.get_document_image_filepath(document_name)
        page_img = mpimg.imread(page_img_path)
        page_img = normalize_image(page_img)
        return page_img

    def __export_bounds(self, document_name, bounds, filename_basepath):
        document_mat = self.__document_image_mat(document_name)
        bounds_ul = np.array(bounds[0]) - np.array([3, 3])
        bounds_lr = np.array(bounds[1]) + np.array([4, 4])
        p_mat = document_mat[bounds_ul[1]:bounds_lr[1], bounds_ul[0]: bounds_lr[0]]

        file_path = filename_basepath + '.png'
        mpimg.imsave(file_path, p_mat, cmap=cm.get_cmap('Greys_r'))


class GTVisualization(object):

    def __init__(self, config):
        self.__config = config

    def visualize_gt(self, query_document, query_word, linecolor='k',
                     linewidth=1.0, plot_label=False, ax=None):

        # page_img = np.flipud(page_img)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            page_img_path = self.__config.get_document_image_filepath(query_document)
            page_img = mpimg.imread(page_img_path)
            ax.imshow(page_img, cmap=cm.get_cmap('Greys_r'))

        ax.autoscale(enable=False)
        ax.set_xticks([])
        ax.set_yticks([])


        gt_reader = GroundTruthReader(base_path=self.__config.data_base,
                                      gtp_encoding=self.__config.gtp_encoding)
        gt_list = gt_reader.read_ground_truth(query_document)
        gt_list_query = [gt_item for gt_item in gt_list if gt_item[0] == query_word]

        for gt_word, gt_coord in gt_list_query:
            ul_x = gt_coord[0][0]
            ul_y = gt_coord[0][1]
            lr_x = gt_coord[1][0]
            lr_y = gt_coord[1][1]

            patch_width = lr_x - ul_x
            patch_height = lr_y - ul_y
            r = matplotlib.patches.Rectangle((ul_x, ul_y),
                                             patch_width, patch_height,
                                             fill=False, edgecolor=linecolor,
                                             linewidth=linewidth)
            if plot_label:
                ax.text(ul_x + 5, lr_y - 5, gt_word, fontsize=6, alpha=0.5)
            ax.add_patch(r)


class PatchVisualization(object):

    @staticmethod
    def visualize_document(config, document, return_img_shape=False, 
                           full_res=False):
        page_img_path = config.get_document_image_filepath(document)
        page_img = mpimg.imread(page_img_path)
        if full_res:
            dpi=80
            # Obtain first to values (in case of RGB image --> channels)
            height, width  = page_img.shape[:2]
                
            # What size does the figure need to be in inches to fit the image?
            figsize = width / float(dpi), height / float(dpi)
    
            # Create a figure of the right size with one axes that 
            # takes up the full figure
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([0, 0, 1, 1])
    
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        # Hide spines, ticks, etc.
        ax.axis('off')
        ax.imshow(page_img, cmap=cm.get_cmap('Greys_r'))

        if return_img_shape:
            return ax, page_img.shape
        else:
            return ax

    @staticmethod
    def visualize_patches(patch_list, ax):

        ax.autoscale(enable=False)
        ax.set_xticks([])
        ax.set_yticks([])

        for bounds, fill, alpha, color, linewidth in patch_list:
            ul_x = bounds[0][0]
            ul_y = bounds[0][1]
            lr_x = bounds[1][0]
            lr_y = bounds[1][1]

            patch_width = lr_x - ul_x
            patch_height = lr_y - ul_y
            r = matplotlib.patches.Rectangle((ul_x, ul_y),
                                             patch_width, patch_height,
                                             fill=fill, edgecolor=color,
                                             facecolor=color, alpha=alpha,
                                             linewidth=linewidth)
            ax.add_patch(r)

    @staticmethod
    def visualize_bounds_labeled(bounds_list, label_list, ax):

        if len(bounds_list) != len(label_list):
            raise ValueError('bounds_list and label_list mismatch!')

        ax.autoscale(enable=False)
        ax.set_xticks([])
        ax.set_yticks([])

        fill = True
        alpha = 0.3
        linewidth = 2.0

        color_map = cm.get_cmap('jet')
        for idx, (bounds, label) in enumerate(zip(bounds_list, label_list)):
            color_idx = idx / float(len(bounds_list))
            color = color_map(color_idx)

            ul_x = int(bounds[0][0])
            ul_y = int(bounds[0][1])
            lr_x = int(bounds[1][0])
            lr_y = int(bounds[1][1])

            patch_width = lr_x - ul_x
            patch_height = lr_y - ul_y
            r = matplotlib.patches.Rectangle((ul_x, ul_y),
                                             patch_width, patch_height,
                                             fill=fill, edgecolor=color,
                                             facecolor=color, alpha=alpha,
                                             linewidth=linewidth)
            ax.text(ul_x + 5, lr_y - 5, label, fontsize=7, alpha=0.7)
            ax.add_patch(r)


class RetrievalVisualization(object):
    '''
    Functions for visualizing retrieval results...
    '''

    def __init__(self, config, document_name):
        '''
        Constructor
        '''
        self.__config = config
        self.__document_name = document_name


    def visualize_retrieval_mat(self, retrieval_mat, plt_score_values=True, fig=None):
        '''

        '''


        page_img_path = self.__config.get_document_image_filepath(self.__document_name)
        page_img = mpimg.imread(page_img_path)
        # page_img = np.flipud(page_img)

        if fig is None:
            fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.imshow(page_img, cmap=cm.get_cmap('Greys_r'))

        ax.autoscale(enable=False)
        ax.set_xticks([])
        ax.set_yticks([])

        patch_score_vector = np.array(retrieval_mat[0, :])

        # Normalize to [0,1]
        # --> 0 best, 1 worst score
        if retrieval_mat.shape[1] > 0:
            min_score = patch_score_vector.min()
            patch_score_vector -= min_score
            max_score = patch_score_vector.max()
            if max_score != 0:
                patch_score_vector /= max_score
            # --> 1 best, 0 worst score
            patch_score_vector -= 1
            patch_score_vector *= -1

        ryg_map = cm.get_cmap('jet')

        for index in range(retrieval_mat.shape[1]):
            patch_score = patch_score_vector[index]
            relevant = retrieval_mat[1, index]
            ul_x = retrieval_mat[2, index]
            ul_y = retrieval_mat[3, index]
            lr_x = retrieval_mat[4, index]
            lr_y = retrieval_mat[5, index]
            patch_width = lr_x - ul_x
            patch_height = lr_y - ul_y
            color = ryg_map(patch_score)
            if relevant == 1:
                r = matplotlib.patches.Rectangle((ul_x, ul_y), patch_width,
                                                 patch_height, fill=True,
                                                 alpha=0.5, linewidth=2.0,
                                                 edgecolor=color, facecolor=color)
            else:
                r = matplotlib.patches.Rectangle((ul_x, ul_y), patch_width,
                                                 patch_height, fill=False,
                                                 edgecolor=color, linewidth=2.0)
            if plt_score_values:
                ax.text(ul_x + 5, ul_y + 10, '%g' % retrieval_mat[0, index],
                        fontsize=6, alpha=0.5)
#                         bbox={'facecolor':'white', 'edgecolor':'white', 'alpha':0.5, 'pad':0})
            ax.add_patch(r)


class ScoreVisualization(object):

    def __init__(self, config, score_cmap_id='jet'):
        '''
        '''

        self.__config = config
        self.__score_cmap_id = score_cmap_id

    def visualize_score_mat(self, document_name, score_mat, score_mat_bounds,
                            normalize_scores=True, fig=None):

        if score_mat_bounds is None or len(score_mat_bounds) != 4:
            raise ValueError('score_mat_bounds undefined, tuple (x_min,y_min,x_max,y_max) required')

        score_mat_extent = (score_mat_bounds[0], score_mat_bounds[2],
                            score_mat_bounds[3], score_mat_bounds[1])


        if normalize_scores:

            # Remove undefined HMM scores
            undef_score = np.max(score_mat) + 1
            score_mat[score_mat == -1] = undef_score

            # Normalize to [0,1]
            # --> 0 best, 1 worst score
            min_score = np.min(score_mat)
            score_mat -= min_score
            max_score = np.max(score_mat)
            if max_score != 0:
                score_mat /= max_score
            # --> 1 best, 0 worst score
            score_mat -= 1
            score_mat *= -1


        page_img_path = self.__config.get_document_image_filepath(document_name)
        page_img = mpimg.imread(page_img_path)
        # page_img = np.flipud(page_img)

        score_mat_shape = (score_mat_bounds[3] - score_mat_bounds[1],
                           score_mat_bounds[2] - score_mat_bounds[0])

        score_mat = scipy.misc.imresize(score_mat, score_mat_shape,
                                        interp='nearest')

        if fig is None:
            fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.imshow(page_img, cmap=cm.get_cmap('Greys_r'))
        ax.autoscale(enable=False)
        ax.imshow(score_mat, cmap=cm.get_cmap(self.__score_cmap_id),
                  alpha=0.75, extent=score_mat_extent)
        ax.set_xticks([])
        ax.set_yticks([])

class ResultPlotter(object):

    @staticmethod
    def bar_plot(y, y_offset, color, x_label=None, y_label=None, x_ticklabels=None, \
                 y_tickspacing=None, y_scale=None, ax_title=None, ax_grid=False, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if y_scale is not None:
            ax.set_yscale(y_scale)

        x = np.arange(1, len(y) + 1)
        bar_width = 0.9
        ax.bar(x - (bar_width / 2.0), y, width=bar_width, color=color)



        if y_tickspacing is not None:
            yticks = np.arange(0, max(y) + 1, y_tickspacing)
            ax.set_yticks(yticks)
            ax.set_yticklabels(tuple(yticks))
            ax.set_ylim(0, max(y) + y_offset)

        if y_label is not None:
            ax.set_ylabel(y_label)

        if x_label is not None:
            ax.set_xlabel(x_label)

        if x_ticklabels is not None:
            ax.set_xticks(np.arange(x.min(), x.max() + 1))
            ax.set_xticklabels(tuple(x_ticklabels))
            ax.set_xlim(0, len(y) + 1)


        if ax_title is not None:
            ax.set_title(ax_title)

        if ax_grid:
            ax.grid()

        return ax

    @staticmethod
    def multi_plot(y_list, colors_list, marker_list, marker_shift_list, legend_list, x_labels=None, ax_title=None, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        p_list = []
        for y, color, marker, marker_shift in zip(y_list, colors_list, marker_list, marker_shift_list):
            x = np.arange(1, len(y) + 1) + marker_shift
            p = ax.plot(x, y, color=color, marker=marker, markersize=10, linestyle='--', linewidth=2)
            p_list.append(p)

        ax.set_yscale('linear')
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_ylim(0, 110)
        ax.set_ylabel('Mean Average Precision (%)')
        if x_labels is not None:
            median_bar_idx = len(y_list) / 2 + 1
            ax.set_xticks(np.arange(1, len(y_list[median_bar_idx]) + 1))
            ax.set_xticklabels(tuple(x_labels))
            ax.set_xlim(0, 16)
            ax.set_xlabel('Number of characters')

        if ax_title is not None:
            ax.set_title(ax_title)
        ax.legend(tuple(p_list), tuple(legend_list), loc=0, fancybox=True, handlelength=3)
        return ax

def plot_quantization(frames, labels, codebook_size, marker_radius=5, alpha=1, ax=None):

    if ax is None:
        ax = plt.gca()
    colormap = cm.get_cmap('jet')
    for (x, y), label in zip(frames[:2, :].T, labels):
        color = colormap(label / float(codebook_size))
        circle = Circle((x, y), radius=marker_radius, fc=color,
                        ec=color, alpha=alpha)
        ax.add_patch(circle)

