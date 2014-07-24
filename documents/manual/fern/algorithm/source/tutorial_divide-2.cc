#include "fern/core/vector_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    std::vector<double> value1 = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> value2 = { 5.0, 4.0, 3.0, 2.0, 1.0 };
    std::vector<double> result(value1.size());

    fern::algebra::divide(fern::sequential, value1, value2, result);
}
