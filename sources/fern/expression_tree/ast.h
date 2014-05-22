#pragma once
#include <boost/variant/recursive_wrapper.hpp>
#include "fern/expression_tree/data.h"
#include "fern/expression_tree/implementation.h"


namespace fern {
namespace expression_tree {

template<
    class T>
struct Operation;


// One of the expressions.
using Expression = boost::variant<
    Data,
    boost::recursive_wrapper<Operation<Constant<int32_t>>>,
    boost::recursive_wrapper<Operation<Constant<int64_t>>>,
    boost::recursive_wrapper<Operation<Constant<double>>>,
    boost::recursive_wrapper<Operation<Raster<int32_t>>>,
    boost::recursive_wrapper<Operation<Raster<int64_t>>>,
    boost::recursive_wrapper<Operation<Raster<double>>>
>;


template<
    class Result>
struct Operation
{

    using result_type = Result;

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

} // namespace expression_tree
} // namespace fern
