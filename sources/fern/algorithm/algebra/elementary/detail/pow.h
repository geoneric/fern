#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/core/result_type.h"


namespace fern {
namespace algorithm {
namespace pow {
namespace detail {

template<
    class Value1,
    class Value2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value1)
    FERN_STATIC_ASSERT(std::is_same, Value2, Value1)

    template<
        class R>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        R& result) const
    {
        errno = 0;
        result = std::pow(static_cast<R>(value1), static_cast<R>(value2));
    }

};


template<
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void pow(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm, OutOfDomainPolicy, OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy,
        execution_policy,
        value1, value2, result);
}

} // namespace detail
} // namespace pow
} // namespace algorithm
} // namespace fern
