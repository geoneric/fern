#pragma once
#include <iostream>
#include "fern/feature/core/masked_constant.h"
#include "fern/algorithm/algebra/elementary/equal.h"
#include "fern/algorithm/statistic/count.h"


template<
    class ExecutionPolicy,
    class Value>
bool equal(
    ExecutionPolicy const& execution_policy,
    Value const& value1,
    Value const& value2)
{
    auto equal_result = fern::clone<int>(value1, 0);

    fern::algebra::equal(execution_policy, value1, value2, equal_result);

    size_t nr_equal_values;

    fern::statistic::count(execution_policy, equal_result, 1, nr_equal_values);

    BOOST_CHECK_EQUAL(nr_equal_values, fern::size(value1));

    return nr_equal_values == fern::size(value1);
}


template<
    class ExecutionPolicy,
    class Value,
    size_t nr_dimensions>
bool equal(
    ExecutionPolicy const& execution_policy,
    fern::MaskedArray<Value, nr_dimensions> const& value1,
    fern::MaskedArray<Value, nr_dimensions> const& value2)
{
    bool values_are_the_same = equal(execution_policy,
            dynamic_cast<fern::Array<Value, nr_dimensions> const&>(value1),
            dynamic_cast<fern::Array<Value, nr_dimensions> const&>(value2));
    bool mask_is_the_same = equal(execution_policy,
            value1.mask(),
            value2.mask());

    BOOST_CHECK(values_are_the_same);
    BOOST_CHECK(mask_is_the_same);

    return values_are_the_same && mask_is_the_same;
}


namespace fern {

template<
    class T>
inline std::ostream& operator<<(
    std::ostream& stream,
    fern::MaskedConstant<T> const& constant)
{
    stream << constant.value() << (constant.mask() ? "(masked)" : "");
    return stream;
}

}
