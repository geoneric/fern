#include "edsl.h"

template<
    typename Expression>
void evaluate(
    Expression const& expression)
{
    calculator_context context;
    boost::proto::eval(expression, context);
}


void main()
{
    // Raster<int> raster1;
    // Raster<int> raster2;

    // auto expression = raster1 + raster2;

    // evaluate(expression);
};
