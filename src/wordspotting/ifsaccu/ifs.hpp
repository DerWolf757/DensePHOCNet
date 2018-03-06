#ifndef IFS_HPP
#define IFS_HPP
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
//#define PY_ARRAY_UNIQUE_SYMBOL PySVQVectors
// import_array will not be called, here. See py_svq_vectors
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <tuple>
#include <boost/python.hpp>

#include <esmeralda/throw_assert.hpp>
#include <esmeralda/svq/svq_vectors.hpp>

class IFS {

public:
	IFS(int x_offset, boost::python::list & x_pos_list) :
			x_offset(x_offset) {
		int n_x_pos_list = boost::python::len(x_pos_list);
		this->x_pos_vec = new std::vector<int>(n_x_pos_list);
		for (int i = 0; i < n_x_pos_list; i++) {
			int x_pos = boost::python::extract<int>(x_pos_list[i]);
			this->x_pos_vec->at(i) = x_pos;
		}
	}
	;
	virtual ~IFS() {
		delete this->x_pos_vec;
	}
	;

	PyArrayObject * ifs_height_arr(
			boost::python::list & y_pos_list, boost::python::list & svq_list) {



		int n_y_pos_list = boost::python::len(y_pos_list);
		int n_svq_list = boost::python::len(svq_list);
		throw_assert(n_y_pos_list == n_svq_lists);

		for (int i = 0; i < n_y_pos_list; i++) {
			int y_pos = boost::python::extract<int>(y_pos_list[i]);
			SVQVectorsView & svq = boost::python::extract<SVQVectorsView&>(
					svq_list[i]);

			unsigned int n_svectors = svq.size();
			unsigned int n_x_pos_vec = this->x_pos_vec->size();
			throw_assert(n_svectors == n_x_pos_vec);

			for (unsigned int j = 0; j < n_svectors; j++) {
				int x_pos = this->x_pos_vec->at(j);
				mx_scorelist_t * scorelist = svq.get_scorelist(j);
				int n_scores = scorelist->n_scores;
				for (int k = 0; k < n_scores; k++) {
					int m_comp_idx = scorelist->score[k].id;
					float m_comp_score = scorelist->score[k].score;
					std::tuple<float, float, float>()
				}
			}

		}

	}
	;
private:
	int x_offset;
	std::vector<int> * x_pos_vec;

}
;

#endif // IFS_HPP
