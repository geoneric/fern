#include "fern/python_extension/algorithm/core/add.h"
#include "fern/algorithm/algebra/elementary/add.h"


namespace fern {
namespace python {

double add(
    double value1,
    double value2)
{
    double result;
    algorithm::algebra::add(algorithm::parallel, value1, value2, result);
    return result;
}

} // namespace python
} // namespace fern
