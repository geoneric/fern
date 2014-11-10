#include "fern/python_extension/algorithm/gdal/algorithm.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/core/macro.h"
#include "fern/python_extension/algorithm/gdal/overloads.h"


namespace fern {
namespace python {

BINARY_ALGORITHM(add)
UNARY_ALGORITHM(slope)

} // namespace python
} // namespace fern
