#pragma once
#include <boost/variant/recursive_wrapper.hpp>
#include "fern/expression_tree/data.h"
#include "fern/expression_tree/implementation.h"


namespace fern {

template<
    class T>
struct Operation;


// One of the expressions.
typedef boost::variant<
    Data,
    boost::recursive_wrapper<Operation<Constant<int32_t>>>,
    boost::recursive_wrapper<Operation<Constant<int64_t>>>,
    boost::recursive_wrapper<Operation<Constant<double>>>,
    boost::recursive_wrapper<Operation<Array<int32_t>>>,
    boost::recursive_wrapper<Operation<Array<int64_t>>>,
    boost::recursive_wrapper<Operation<Array<double>>>
> Expression;


template<
    class Result>
struct Operation
{

    typedef Result ResultType;

    Operation(
        std::string const& name,
        Implementation const& implementation,
        std::vector<Expression> const& expressions)

        : name(name),
          implementation(implementation),
          expressions(expressions)

    {
    }

    std::string name;

    Implementation implementation;

    std::vector<Expression> expressions;

};

} // namespace fern
