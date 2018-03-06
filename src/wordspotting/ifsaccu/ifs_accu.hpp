#ifndef IFS_ACCU_HPP
#define IFS_ACCU_HPP
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
#include <boost/python.hpp>

#include <esmeralda/throw_assert.hpp>

class IFSAccu {

public:
	IFSAccu(int vw_bounds_ul_x, int vw_bounds_ul_y, int cell_size_x,
			int cell_size_y, int query_width, int query_center_ref) :
			vw_bounds_ul_x(vw_bounds_ul_x), vw_bounds_ul_y(vw_bounds_ul_y), cell_size_x(
					cell_size_x), cell_size_y(cell_size_y), query_width(
					query_width), query_center_ref(query_center_ref) {
	}
	;
	virtual ~IFSAccu() {
	}
	;

	void accumulate_ght(boost::python::list & ifs_list,
			PyArrayObject & cell_vw_arr, PyArrayObject & accu_arr) {

		int vw_bounds_ul_x = this->vw_bounds_ul_x;
		int vw_bounds_ul_y = this->vw_bounds_ul_y;
		int cell_size_x = this->cell_size_x;
		int cell_size_y = this->cell_size_y;
		int query_width = this->query_width;
		int query_center_ref = this->query_center_ref;
		throw_assert(PyArray_NDIM(&cell_vw_arr) == 2);
		throw_assert(PyArray_TYPE(&cell_vw_arr) == NPY_FLOAT32);
		const int cell_vw_arr_dim0 = PyArray_DIM(&cell_vw_arr, 0);
		const int cell_vw_arr_dim1 = PyArray_DIM(&cell_vw_arr, 1);
		throw_assert(PyArray_NDIM(&accu_arr) == 2);
		throw_assert(PyArray_TYPE(&accu_arr) == NPY_FLOAT32);
		const int accu_arr_dim0 = PyArray_DIM(&accu_arr, 0);
		const int accu_arr_dim1 = PyArray_DIM(&accu_arr, 1);
		// Cast to float for floating-point division
		float n_cells = static_cast<float>(cell_vw_arr_dim0);
		float cell_area_center = query_width / (2.0 * n_cells);
		for (int c_idx = 0; c_idx < cell_vw_arr_dim0; c_idx++) {
			// c_idx is the index of the current cell
			// Set x voting offset for current cell
			int x_offset = static_cast<int>(query_center_ref
					- (((c_idx / n_cells) * query_width) + cell_area_center));
			for (int vw_idx = 0; vw_idx < cell_vw_arr_dim1; vw_idx++) {
				// Extract voting-mass for current vw
				float cell_prob = *static_cast<float*>(PyArray_GETPTR2(
						&cell_vw_arr, c_idx, vw_idx));
				// Skip vw if there is no voting mass
				if (cell_prob == 0)
					continue;

				// Extract xys position/score array for current vw from IFS
//				std::cout << "xy_arr for vw: " << vw << std::endl;
				PyArrayObject * xys_arr =
						boost::python::extract<PyArrayObject*>(
								ifs_list[vw_idx]);
				throw_assert(PyArray_TYPE(xys_arr) == NPY_FLOAT32);
				const int xys_arr_dim0 = PyArray_DIM(xys_arr, 0);
				// Process xys position/score array
				for (int p = 0; p < xys_arr_dim0; p++) {
					float x = *static_cast<float*>(PyArray_GETPTR2(xys_arr, p,
							0));
					float y = *static_cast<float*>(PyArray_GETPTR2(xys_arr, p,
							1));
					float frame_prob = *static_cast<float*>(PyArray_GETPTR2(
							xys_arr, p, 2));
					float score = cell_prob * frame_prob;
					// Compute integer divison (floor for accu cell coords)
					int c_accu = ((x + x_offset - vw_bounds_ul_x) / cell_size_x);
					int r_accu = (y - vw_bounds_ul_y) / cell_size_y;
					if (r_accu >= 0 && c_accu >= 0 && r_accu < accu_arr_dim0
							&& c_accu < accu_arr_dim1) {
						*static_cast<float*>(PyArray_GETPTR2(&accu_arr, r_accu,
								c_accu)) += score;
					}
				}
			}
		}
	}
	;
private:
	int vw_bounds_ul_x;
	int vw_bounds_ul_y;
	int cell_size_x;
	int cell_size_y;
	int query_width;
	int query_center_ref;

}
;

#endif // IFS_ACCU_HPP
