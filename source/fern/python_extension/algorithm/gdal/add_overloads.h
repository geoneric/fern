#pragma once
#include "fern/python_extension/algorithm/core/binary_operation_map.h"
#include "fern/python_extension/algorithm/gdal/wrapped_data_type.h"


namespace fern {
namespace python {
namespace gdal {

using BinaryAlgorithmKey = std::tuple<WrappedDataType, WrappedDataType>;
extern BinaryOperationMap<BinaryAlgorithmKey> add_overloads;

} // namespace gdal
} // namespace python
} // namespace fern
