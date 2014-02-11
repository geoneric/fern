#pragma once
#include <type_traits>


namespace fern {
namespace expression_tree {

//! Wrapper class for constant values.
/*!
*/
template<
    class Result>
struct Constant
{

    static_assert(std::is_arithmetic<Result>::value, "Type must be arithmetic");

    Constant()
        : value()
    {
    }

    Constant(
        Result const& value)
        : value(value)
    {
    }

    // Only allow explicit conversion from constant to int. Otherwise we run
    // into ambiguity when functions are overloaded for Constant and Result.
    explicit operator Result()
    {
        return value;
    }

    typedef Result value_type;

    typedef Constant<Result> result_type;

    Result value;

};

} // namespace expression_tree
} // namespace fern
