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
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/unary_local_operation.h"


namespace fern {
namespace algorithm {
namespace sqrt {
namespace detail {

template<
    typename Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

    template<
        typename Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = std::sqrt(value);
    }

};


// http://www.codeproject.com/Articles/69941/Best-Square-Root-Method-Algorithm-Function-Precisi


// #define SQRT_MAGIC_F 0x5f3759df 
// double sqrt2(const double x)
// {
//   const double xhalf = 0.5f*x;
//  
//   union // get bits for double value
//   {
//     double x;
//     int i;
//   } u;
//   u.x = x;
//   u.i = SQRT_MAGIC_F - (u.i >> 1);  // gives initial guess y0
//   return x*u.x*(1.5f - xhalf*u.x*u.x);// Newton step, repeating increases accuracy 
// }


// template<>
// struct Algorithm<
//     float>
// {
// 
//     template<
//         typename Result>
//     inline void operator()(
//         float const& value,
//         Result& result) const
//     {
//         FERN_STATIC_ASSERT(std::is_same, Result, float)
// 
//         _mm_store_ss( &result, _mm_sqrt_ss( _mm_load_ss( &value ) ) );
//     }
// 
// };


template<
    template<typename> class OutOfDomainPolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void sqrt(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        OutOfDomainPolicy, unary::DiscardRangeErrors>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace sqrt
} // namespace algorithm
} // namespace fern
