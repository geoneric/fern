#pragma once
#include <utility>
#include "fern/example/algorithm/operation.h"


namespace example {

template<
    typename Value1,
    typename Value2>
fern::algorithm::add::result_type<Value1, Value2> operator+(
    Value1 const& lhs,
    Value2 const& rhs)
{
    return std::move(add(lhs, rhs));
}

} // namespace example
