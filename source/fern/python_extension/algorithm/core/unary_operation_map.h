#pragma once
#include <functional>
#include <map>
#include "fern/python_extension/algorithm/gdal/wrapped_data_type.h"


namespace fern {
namespace python {

using UnaryAlgorithmKey = WrappedDataType;
using UnaryAlgorithm = std::function<PyObject*(PyObject*)>;
using UnaryOperationMap = std::map<UnaryAlgorithmKey, UnaryAlgorithm>;

} // namespace python
} // namespace fern
