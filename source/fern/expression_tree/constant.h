// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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

    using value_type = Result;

    using result_type = Constant<Result>;

    Result value;

};

} // namespace expression_tree
} // namespace fern
