#include "fern/python_extension/algorithm/numpy/algorithm.h"
#include "fern/python_extension/core/error.h"
#include "fern/python_extension/algorithm/core/macro.h"
#include "fern/python_extension/algorithm/numpy/overloads.h"


namespace fern {
namespace python {
namespace numpy {

BINARY_ALGORITHM(add)
UNARY_ALGORITHM(sqrt)

} // namespace numpy
} // namespace python
} // namespace fern

