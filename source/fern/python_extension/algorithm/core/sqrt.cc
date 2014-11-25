#include "fern/python_extension/algorithm/core/sqrt.h"
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/sqrt.h"


namespace fern {
namespace python {
namespace core {

double sqrt(
    double value)
{
    double result;
    algorithm::algebra::sqrt(algorithm::parallel, value, result);
    return result;
}

} // namespace core
} // namespace python
} // namespace fern
