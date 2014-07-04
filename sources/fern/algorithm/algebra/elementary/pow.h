#pragma once
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/pow.h"


namespace fern {
namespace pow {

template<
    class Base,
    class Exponent
>
class OutOfDomainPolicy
{

public:

    inline static bool within_domain(
        Base const& base,
        Exponent const& exponent)
    {
        if(base < Base(0)) {
            Base integral, fractional;
            fractional = std::modf(exponent, &integral);

            if(fractional != Base(0)) {
                return false;
            }
        }
        else if(base == Base(0) && exponent < Exponent(0)) {
            return false;
        }

        return true;
    }

};


template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = DetectOutOfRangeByErrno<Value1, Value2, Result>;

} // namespace pow


namespace algebra {

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
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value1>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value2>, value_type<Value1>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value1>)

    pow::detail::pow<OutOfDomainPolicy, OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


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
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    pow<OutOfDomainPolicy, OutOfRangePolicy>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value1, value2, result);
}


template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void pow(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    pow<binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
        InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value1, value2, result);
}

} // namespace algebra
} // namespace fern
