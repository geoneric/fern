#pragma once
#include <functional>
#include <map>
#include <tuple>
#include "fern/python_extension/algorithm/gdal/wrapped_data_type.h"


namespace fern {
namespace python {

using BinaryAlgorithmKey = std::tuple<WrappedDataType, WrappedDataType>;
using BinaryAlgorithm = std::function<PyObject*(PyObject*, PyObject*)>;
using BinaryOperationMap = std::map<BinaryAlgorithmKey, BinaryAlgorithm>;

} // namespace python
} // namespace fern
