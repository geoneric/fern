#pragma once
#include "fern/python_extension/algorithm/core/unary_operation_map.h"
#include "fern/python_extension/algorithm/numpy/wrapped_data_type.h"


namespace fern {
namespace python {
namespace numpy {

using UnaryAlgorithmKey = WrappedDataType;
extern UnaryOperationMap<UnaryAlgorithmKey> sqrt_overloads;

} // namespace numpy
} // namespace python
} // namespace fern
