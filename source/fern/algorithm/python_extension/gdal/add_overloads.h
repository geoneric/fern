#pragma once
#include <functional>
#include <map>
#include <tuple>
#include "fern/algorithm/python_extension/gdal/wrapped_data_type.h"


namespace fern {
namespace python {

using PyBinaryAlgorithmKey = std::tuple<WrappedDataType, WrappedDataType>;
using PyBinaryAlgorithm = std::function<PyObject*(PyObject*, PyObject*)>;

extern std::map<PyBinaryAlgorithmKey, PyBinaryAlgorithm> add_overloads;

} // namespace python
} // namespace fern
