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
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/core/result_type.h"


namespace fern {
namespace algorithm {
namespace divide {
namespace detail {
namespace dispatch {

template<
    typename Value1,
    typename Value2,
    typename R,
    typename A1NumberCategory,
    typename A2NumberCategory>
struct within_range
{
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // unsigned / signed
        return true;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // signed / signed
        return true;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    unsigned_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::algorithm::Result<
            Value1, Value2>::type, Result)

        // unsigned / signed
        return true;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::algorithm::Result<
            Value1, Value2>::type, Result)

        // signed / unsigned
        return true;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    integer_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result)
    {
        return within_range<Value1, Value2, Result,
            typename TypeTraits<Value1>::number_category,
            typename TypeTraits<Value2>::number_category>::calculate(value1,
                value2, result);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    floating_point_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // float / float
        return std::isfinite(result);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // integral / float
        return std::isfinite(result);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    floating_point_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* values2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // float / integral
        return std::isfinite(result);
    }
};

} // namespace dispatch


template<
    typename Value1,
    typename Value2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Value2)

    template<
        typename R>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        R& result) const
    {
        assert(value2 != Value2(0));
        result = static_cast<R>(value1) / static_cast<R>(value2);
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
void divide(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm,
        OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value1, value2, result);
}

} // namespace detail
} // namespace divide
} // namespace algorithm
} // namespace fern
