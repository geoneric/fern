#pragma once
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/unite_no_data.h"


namespace fern {
namespace algorithm {
namespace core {

//! Unite the no-data of \a value1 and \a value2 and store the results in result.
/*!
    \ingroup       core
    \sa            @ref fern_algorithm_core

    Uniting no-data means that if an element from \a value1 or \a value2
    contains a no-data, \a result will contain a no-data. This is a useful
    operation when preparing the inputs for other algorithms. A lot of
    algorithms assume that the result already contains no-data for those
    elements for which one of the inputs contains a no-data.

    \a value1, \a value2 and \a result are not used. They may, of course,
    be used by the no-data policies passed in.

    The value types of \a value1, \a value2 and \a result are not relevant.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    unite_no_data::detail::unite_no_data<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    \ingroup       core
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    unite_no_data<>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
    \ingroup       core
    \overload
*/
template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<SkipNoData<>, SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    unite_no_data<>(InputNoDataPolicy(SkipNoData<>(), SkipNoData<>()),
        output_no_data_policy, execution_policy, value1, value2, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern
