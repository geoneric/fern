#pragma once
#include <tuple>
#include "fern/python_extension/algorithm/core/binary_operation_map.h"
#include "fern/python_extension/algorithm/numpy/wrapped_data_type.h"


namespace fern {
namespace python {
namespace numpy {

using BinaryAlgorithmKey = std::tuple<WrappedDataType, WrappedDataType>;
extern BinaryOperationMap<BinaryAlgorithmKey> add_overloads;

} // namespace numpy
} // namespace python
} // namespace fern
