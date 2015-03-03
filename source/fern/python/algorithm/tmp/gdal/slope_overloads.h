#pragma once
#include "fern/python_extension/algorithm/core/unary_operation_map.h"
#include "fern/python_extension/algorithm/gdal/wrapped_data_type.h"


namespace fern {
namespace python {
namespace gdal {

using UnaryAlgorithmKey = WrappedDataType;
extern UnaryOperationMap<UnaryAlgorithmKey> slope_overloads;

} // namespace gdal
} // namespace python
} // namespace fern
