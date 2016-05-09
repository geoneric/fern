#pragma once
#include <boost/proto.hpp>

namespace edsl {

template<
    int I>
struct placeholder
{
};


boost::proto::terminal<placeholder<0>>::type const _1 = {{}};
boost::proto::terminal<placeholder<1>>::type const _2 = {{}};


struct calculator_context
  : boost::proto::callable_context< calculator_context const >
{
    // Values to replace the placeholders
    std::vector<double> args;

    // Define the result type of the calculator.
    // (This makes the calculator_context "callable".)
    typedef double result_type;

    // Handle the placeholders:
    template<int I>
    double operator()(
        boost::proto::tag::terminal, placeholder<I>) const
    {
        return this->args[I];
    }
};

} // namespace edsl
