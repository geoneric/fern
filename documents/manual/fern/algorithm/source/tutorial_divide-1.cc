#include "fern/algorithm/algebra/elementary/divide.h"


int main()
{
    namespace fa = fern::algorithm;

    double value1 = 1.0;
    double value2 = 2.0;
    double result;

    fa::algebra::divide(fa::sequential, value1, value2, result);
}
