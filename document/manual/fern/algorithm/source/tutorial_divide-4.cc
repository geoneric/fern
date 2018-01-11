#include <cstdlib>
#include "fern/core/data_customization_point/vector.h"
#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    namespace fa = fern::algorithm;

    fa::ParallelExecutionPolicy parallel;
    std::vector<double> value1 = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    std::vector<double> value2 = { 5.0, 4.0, 3.0, 2.0, 1.0 };
    std::vector<double> result(value1.size());

    fa::algebra::divide(parallel, value1, value2, result);

    return EXIT_SUCCESS;
}
