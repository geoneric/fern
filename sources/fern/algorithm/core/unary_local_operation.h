#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/collection_traits.h"
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
} // namespace fern
