#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#define PY_ARRAY_UNIQUE_SYMBOL PyIFSAccu
// import_array will not be called, here. See py_calculator
#include <numpy/arrayobject.h>
#include <boost/python.hpp>
#include <string>
#include "ifs_accu.hpp"

using namespace boost;
using namespace boost::python;

void* extract_pyarray(PyObject* x) {
	return x;
}

void translate(const std::runtime_error& e) {
	PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE(_ifs_accu) {
	class_<IFSAccu>("IFSAccu", init<int,int,int,int,int,int>())
	.def("accumulate_ght",&IFSAccu::accumulate_ght)
	;

	converter::registry::insert(
			&extract_pyarray, type_id<PyArrayObject>());

	register_exception_translator<std::runtime_error> (translate);

	import_array();
}
