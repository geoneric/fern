#include "fern/python_extension/algorithm/core/add.h"
#include "fern/core/constant_traits.h"
#include "fern/algorithm/algebra/elementary/add.h"


namespace fern {
namespace python {
namespace core {

int64_t add(
    int64_t value1,
    int64_t value2)
{
    int64_t result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
    return result;
}


float64_t add(
    float64_t value1,
    float64_t value2)
{
    float64_t result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
    return result;
}


float64_t add(
    float64_t value1,
    int64_t value2)
{
    float64_t result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
    return result;
}


float64_t add(
    int64_t value1,
    float64_t value2)
{
    float64_t result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
    return result;
}

} // namespace core
} // namespace python
} // namespace fern
