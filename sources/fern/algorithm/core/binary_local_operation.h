#pragma once
#include <type_traits>
#include <utility>
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/collection_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace detail {
namespace dispatch {

template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm,
    class A1CollectionCategory,
    class Values2CollectionCategory>
struct BinaryLocalOperation
{
};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        constant_tag,
        constant_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + constant
    inline void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        using INDP = InputNoDataPolicy;

        if(!INDP::is_no_data()) {
            _algorithm(get(values1), get(values2), get(result));
        }
    }

private:

    Algorithm      _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_1d_tag,
        array_1d_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + collection
    void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1) == fern::size(values2));
        assert(fern::size(values1) == fern::size(result));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const size = fern::size(values1);

        for(size_t i = 0; i < size; ++i) {
            if(!INDP::is_no_data(i)) {
                const_reference<Values1> a1(fern::get(values1, i));
                const_reference<Values2> a2(fern::get(values2, i));

                if(!OODP::within_domain(a1, a2)) {
                    ONDP::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OORP::within_range(a1, a2, fern::get(result, i))) {
                    ONDP::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_2d_tag,
        array_2d_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + collection
    void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1, 0) == fern::size(values2, 0));
        assert(fern::size(values1, 1) == fern::size(values2, 1));
        assert(fern::size(values1, 0) == fern::size(result, 0));
        assert(fern::size(values1, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(values1, 0);
        size_t const size2 = fern::size(values1, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values1, values2, result);
    }

    // collection + constant
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1, 0) == fern::size(values2, 0));
        assert(fern::size(values1, 1) == fern::size(values2, 1));
        assert(fern::size(values1, 0) == fern::size(result, 0));
        assert(fern::size(values1, 1) == fern::size(result, 1));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(!INDP::is_no_data(i, j)) {
                    const_reference<Values1> a1(fern::get(values1, i, j));
                    const_reference<Values2> a2(fern::get(values2, i, j));

                    if(!OODP::within_domain(a1, a2)) {
                        ONDP::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OORP::within_range(a1, a2, fern::get(result, i, j))) {
                        ONDP::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        constant_tag,
        array_1d_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + collection
    inline void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values2) == fern::size(result));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const size = fern::size(values2);

        const_reference<Values1> a1(fern::get(values1));

        for(size_t i = 0; i < size; ++i) {
            if(!INDP::is_no_data(i)) {
                const_reference<Values2> a2(fern::get(values2, i));

                if(!OODP::within_domain(a1, a2)) {
                    ONDP::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OORP::within_range(a1, a2, fern::get(result, i))) {
                    ONDP::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        constant_tag,
        array_2d_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + collection
    inline void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values2, 0) == fern::size(result, 0));
        assert(fern::size(values2, 1) == fern::size(result, 1));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const size1 = fern::size(values2, 0);
        size_t const size2 = fern::size(values2, 1);

        const_reference<Values1> a1(fern::get(values1));

        for(size_t i = 0; i < size1; ++i) {
            for(size_t j = 0; j < size2; ++j) {
                if(!INDP::is_no_data(i, j)) {
                    const_reference<Values2> a2(fern::get(values2, i, j));

                    if(!OODP::within_domain(a1, a2)) {
                        ONDP::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OORP::within_range(a1, a2, fern::get(result, i, j))) {
                        ONDP::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_1d_tag,
        constant_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + constant
    inline void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1) == fern::size(result));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const size = fern::size(values1);

        const_reference<Values2> a2(fern::get(values2));

        for(size_t i = 0; i < size; ++i) {
            if(!INDP::is_no_data(i)) {
                const_reference<Values1> a1(fern::get(values1, i));

                if(!OODP::within_domain(a1, a2)) {
                    ONDP::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OORP::within_range(a1, a2, fern::get(result, i))) {
                    ONDP::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class Values1, class Values2, class Result,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Algorithm>
class BinaryLocalOperation<Values1, Values2, Result,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Algorithm,
        array_2d_tag,
        constant_tag>:

    public OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>,
    public OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
        value_type<Result>>,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Values2>)

public:

    BinaryLocalOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(),
          OutputNoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryLocalOperation(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy<value_type<Values1>, value_type<Values2>>(),
          OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
              value_type<Result>>(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + constant
    inline void calculate(
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1, 0) == fern::size(result, 0));
        assert(fern::size(values1, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(values1, 0);
        size_t const size2 = fern::size(values1, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values1, values2, result);
    }

    // collection + constant
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values1 const& values1,
        Values2 const& values2,
        Result& result)
    {
        assert(fern::size(values1, 0) == fern::size(result, 0));
        assert(fern::size(values1, 1) == fern::size(result, 1));

        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;
        using OODP = OutOfDomainPolicy<value_type<Values1>,
            value_type<Values2>>;
        using OORP = OutOfRangePolicy<value_type<Values1>, value_type<Values2>,
            value_type<Result>>;

        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        const_reference<Values2> a2(fern::get(values2));

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(!INDP::is_no_data(i, j)) {
                    const_reference<Values1> a1(fern::get(values1, i, j));

                    if(!OODP::within_domain(a1, a2)) {
                        ONDP::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OORP::within_range(a1, a2, fern::get(result, i, j))) {
                        ONDP::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};

} // namespace dispatch
} // namespace detail
} // namespace fern
