#pragma once
#include <utility>
// Include the relevant traits before including the algorithms.
#include "fern/core/constant_traits.h"
#include "fern/example/algorithm/raster_traits.h"
#include "fern/algorithm/core/cast.h"
#include "fern/algorithm/spatial/focal/slope.h"
#include "fern/algorithm/algebra/elementary/add.h"


namespace example {

// Execution policy to use by the algorithms: sequential or parallel.
extern fern::ExecutionPolicy execution_policy;


template<
    class Value1,
    class Value2>
fern::add::result_type<Value1, Value2> add(
    Value1 const& lhs,
    Value2 const& rhs)
{
    assert(fern::cell_size(lhs, 0) == fern::cell_size(lhs, 1));
    assert(fern::cell_size(lhs, 0) == fern::cell_size(rhs, 0));
    assert(fern::cell_size(lhs, 1) == fern::cell_size(rhs, 1));
    assert(fern::size(lhs, 0) == fern::size(rhs, 0));
    assert(fern::size(lhs, 1) == fern::size(rhs, 1));

    fern::add::result_type<Value1, Value2> result(
        fern::cell_size(lhs, 0), fern::size(lhs, 0), fern::size(lhs, 1));

    fern::algebra::add(execution_policy, lhs, rhs, result);

    return std::move(result);
}


template<
    class ResultValueType,
    class Value>
fern::Collection<Value, ResultValueType> cast(
    Value const& value)
{
    assert(fern::cell_size(value, 0) == fern::cell_size(value, 1));

    fern::Collection<Value, ResultValueType> result(
        fern::cell_size(value, 0),
        fern::size(value, 0),
        fern::size(value, 1));

    fern::core::cast(execution_policy, value, result);

    return std::move(result);
}


template<
    class Value>
fern::Collection<Value, fern::value_type<Value>> slope(
    Value const& value)
{
    assert(fern::cell_size(value, 0) == fern::cell_size(value, 1));

    fern::Collection<Value, fern::value_type<Value>> result(
        fern::cell_size(value, 0), fern::size(value, 0), fern::size(value, 1));

    fern::spatial::slope(execution_policy, value, result);

    return std::move(result);
}

} // namespace example
