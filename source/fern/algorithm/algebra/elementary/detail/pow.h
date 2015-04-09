// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
    typename Value1,
    typename Value2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value1)
    FERN_STATIC_ASSERT(std::is_same, Value2, Value1)

    template<
        typename R>
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
    template<typename, typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void pow(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
