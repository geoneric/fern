#include <cstdlib>
#include "fern/core/data_customization_point/constant.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    namespace fa = fern::algorithm;

    fern::Array<double, 2> value1 = {
        { 1.0, 2.0 },
        { 3.0, 4.0 },
        { 5.0, 6.0 }
    };
    double value2 = 9.0;
    fern::Array<double, 2> result(fern::extents[2][3]);

    fa::algebra::divide(fa::sequential, value1, value2, result);

    return EXIT_SUCCESS;
}
