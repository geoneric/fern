#include <cstdlib>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    namespace fa = fern::algorithm;

    fa::SequentialExecutionPolicy sequential;

    double value1 = 1.0;
    double value2 = 2.0;
    double result;

    fa::algebra::divide(sequential, value1, value2, result);

    return EXIT_SUCCESS;
}
