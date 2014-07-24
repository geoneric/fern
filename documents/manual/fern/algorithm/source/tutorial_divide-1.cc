#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    double value1 = 1.0;
    double value2 = 2.0;
    double result;

    fern::algebra::divide(fern::sequential, value1, value2, result);
}
