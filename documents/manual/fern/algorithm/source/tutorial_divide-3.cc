#include "fern/feature/core/array_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    fern::Array<double, 2> value1 = {
        { 1.0, 2.0 },
        { 3.0, 4.0 },
        { 5.0, 6.0 }
    };
    double value2 = 9.0;
    fern::Array<double, 2> result(fern::extents[2][3]);

    fern::algebra::divide(fern::sequential, value1, value2, result);
}
