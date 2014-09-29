#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/unary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace cast {
namespace detail {
namespace dispatch {

template<
    class Value,
    class R,
    class ValueNumberCategory,
    class ResultNumberCategory>
struct within_range
{
};


#define UNCONDITIONAL_WITHIN_RANGE( \
    number_category1, \
    number_category2) \
template< \
    class Value, \
    class Result> \
struct within_range< \
    Value, \
    Result, \
    number_category1, \
    number_category2> \
{ \
    inline static constexpr bool calculate( \
        Value const&, \
        Result const&) \
    { \
        return true; \
    } \
};

UNCONDITIONAL_WITHIN_RANGE(boolean_tag, boolean_tag)
UNCONDITIONAL_WITHIN_RANGE(boolean_tag, integer_tag)
UNCONDITIONAL_WITHIN_RANGE(boolean_tag, floating_point_tag)

#undef UNCONDITIONAL_WITHIN_RANGE


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        static_assert(sizeof(Value) <= sizeof(Result), "");  // For now.

        return true;
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value const& value,
        Result const& result)
    {
        static_assert(sizeof(value) <= sizeof(result), "");  // For now.

        return true;
    }
};


// TODO signed_integer -> unsigned_integer
// TODO unsigned_integer -> signed_integer


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    integer_tag,
    integer_tag>
{
    inline static bool calculate(
        Value const& value,
        Result const& result)
    {
        return within_range<Value, Result,
            typename TypeTraits<Value>::number_category,
            typename TypeTraits<Result>::number_category>::calculate(
                value, result);
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    unsigned_integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, Value, Result)

        return true;
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    signed_integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, Value, Result)

        return true;
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    floating_point_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, Value, Result)

        return true;
    }
};


// TODO floating_point -> signed_integer
// TODO floating_point -> unsigned_integer


} // namespace dispatch


template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)

    template<
        class R>
    inline void operator()(
        Value const& value,
        R& result) const
    {
        result = static_cast<R>(value);
    }

};


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void cast(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        unary::DiscardDomainErrors, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace cast
} // namespace algorithm
} // namespace fern
