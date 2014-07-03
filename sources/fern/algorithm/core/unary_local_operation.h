#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/collection_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm,
    class ValuesCollectionCategory>
class UnaryLocalOperation
{
};


template<class Values, class Result,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class UnaryLocalOperation<Values, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_0d_tag>:

    public OutOfDomainPolicy<value_type<Values>>,
    public OutOfRangePolicy<value_type<Values>, value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    UnaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    UnaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // f(constant)
    inline void calculate(
        Values const& values,
        Result& result)
    {
        using INDP = InputNoDataPolicy;

        if(!INDP::is_no_data()) {
            _algorithm(get(values), get(result));
        }
    }

private:

    Algorithm      _algorithm;

};


template<class Values, class Result,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class UnaryLocalOperation<Values, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_1d_tag>:

    public OutOfDomainPolicy<value_type<Values>>,
    public OutOfRangePolicy<value_type<Values>, value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values>)

public:

    UnaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    UnaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // f(collection)
    void calculate(
        Values const& values,
        Result& result)
    {
        assert(fern::size(values) == fern::size(result));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values>>;
        using OORP = OutOfRangePolicy<value_type<Values>, value_type<Result>>;

        size_t const size = fern::size(values);

        for(size_t i = 0; i < size; ++i) {
            if(!INDP::is_no_data(i)) {
                const_reference<Values> a(fern::get(values, i));

                if(!OODP::within_domain(a)) {
                    ONDP::mark_as_no_data(i);
                }

                _algorithm(a, fern::get(result, i));

                if(!OORP::within_range(a, fern::get(result, i))) {
                    ONDP::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};


template<class Values, class Result,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class UnaryLocalOperation<Values, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_2d_tag>:

    public OutOfDomainPolicy<value_type<Values>>,
    public OutOfRangePolicy<value_type<Values>, value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values>)

public:

    UnaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    UnaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values>>(),
          OutOfRangePolicy<value_type<Values>, value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // f(collection)
    void calculate(
        Values const& values,
        Result& result)
    {
        assert(fern::size(values, 0) == fern::size(result, 0));
        assert(fern::size(values, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values, result);
    }

    // f(collection)
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        Result& result)
    {
        assert(fern::size(values, 0) == fern::size(result, 0));
        assert(fern::size(values, 1) == fern::size(result, 1));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values>>;
        using OORP = OutOfRangePolicy<value_type<Values>, value_type<Result>>;

        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(!INDP::is_no_data(i, j)) {
                    const_reference<Values> a(fern::get(values, i, j));

                    if(!OODP::within_domain(a)) {
                        ONDP::mark_as_no_data(i, j);
                    }

                    _algorithm(a, fern::get(result, i, j));

                    if(!OORP::within_range(a, fern::get(result, i, j))) {
                        ONDP::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};

} // namespace dispatch
} // namespace detail


namespace detail {
namespace dispatch2 {

template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class UnaryLocalOperation
{
};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_0d_tag>

{

    // f(0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data()) {
            const_reference<Value> v(fern::get(value));

            if(!OutOfDomainPolicy::within_domain(v)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data();
            }
            else {
                Algorithm algorithm;
                reference<Value> r(fern::get(result));

                algorithm(v, r);

                if(!OutOfRangePolicy::within_range(v, r)) {
                    // Result value is out-of-range. Mark result value as
                    // no-data. Result value contains the out-of-range
                    // value (this may be overriden by
                    // output_no_data_policy, depending on its
                    // implementation).
                    output_no_data_policy.mark_as_no_data();
                }
            }
        }
    }

};


template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data(i)) {
            const_reference<Value> v(fern::get(value, i));

            if(!OutOfDomainPolicy::within_domain(v)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Value> r(fern::get(result, i));

                algorithm(v, r);

                if(!OutOfRangePolicy::within_range(v, r)) {
                    // Result value is out-of-range. Mark result value as
                    // no-data. Result value contains the out-of-range
                    // value (this may be overriden by
                    // output_no_data_policy, depending on its
                    // implementation).
                    output_no_data_policy.mark_as_no_data(i);
                }
            }
        }
    }
}


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value) == fern::size(result));

        Algorithm algorithm;

        operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value))}, value, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value) == fern::size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_1d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        size_t const start1 = 0;
        size_t const finish1 = fern::size(result, 0);
        size_t const start2 = 0;
        size_t const finish2 = fern::size(result, 1);
        Algorithm algorithm;

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {

                // Don't do anything if the input value is no-data. We assume
                // that input no-data values are already marked as such in the
                // result.
                if(!input_no_data_policy.is_no_data(i, j)) {
                    const_reference<Value> v(fern::get(value, i, j));

                    if(!OutOfDomainPolicy::within_domain(v)) {
                        // Input value is out of domain. Mark result value as
                        // no-data. Don't change the result value.
                        output_no_data_policy.mark_as_no_data(i, j);
                    }
                    else {
                        reference<Value> r(fern::get(result, i, j));

                        algorithm(v, r);

                        if(!OutOfRangePolicy::within_range(v, r)) {
                            // Result value is out-of-range. Mark result
                            // value as no-data. Result value contains the
                            // out-of-range value (this may be overriden
                            // by output_no_data_policy, depending on its
                            // implementation).
                            output_no_data_policy.mark_as_no_data(i, j);
                        }
                    }
                }
            }
        }
    }

};

} // namespace dispatch2
} // namespace detail


template<
    template<class> class Algorithm,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void unary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    detail::dispatch2::UnaryLocalOperation<
        Algorithm<value_type<Value>>,
        OutOfDomainPolicy<value_type<Value>>,
        OutOfRangePolicy<value_type<Value>, value_type<Result>>,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace fern
