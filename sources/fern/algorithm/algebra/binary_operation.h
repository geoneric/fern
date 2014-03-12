#pragma once
#include <type_traits>
#include <utility>
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/collection_traits.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace detail {
namespace dispatch {

template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct BinaryOperation
{
};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        constant_tag,
        constant_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        _algorithm(argument1, argument2, result);
    }

private:

    Algorithm      _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        array_1d_tag,
        array_1d_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A1>::value_type)
    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A2>::value_type)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + collection
    void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1) == fern::size(argument2));
        assert(fern::size(argument1) == fern::size(result));

        size_t const size = fern::size(argument1);

        for(size_t i = 0; i < size; ++i) {
            if(!NoDataPolicy::is_no_data(i)) {
                typename ArgumentTraits<A1>::value_type a1(fern::get(
                    argument1, i));
                typename ArgumentTraits<A2>::value_type a2(fern::get(
                    argument2, i));

                if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                    NoDataPolicy::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                        result, i))) {
                    NoDataPolicy::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        array_2d_tag,
        array_2d_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A1>::value_type)
    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A2>::value_type)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + collection
    void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1, 0) == fern::size(argument2, 0));
        assert(fern::size(argument1, 1) == fern::size(argument2, 1));
        assert(fern::size(argument1, 0) == fern::size(result, 0));
        assert(fern::size(argument1, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, argument1, argument2, result);
    }

    // collection + constant
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1, 0) == fern::size(argument2, 0));
        assert(fern::size(argument1, 1) == fern::size(argument2, 1));
        assert(fern::size(argument1, 0) == fern::size(result, 0));
        assert(fern::size(argument1, 1) == fern::size(result, 1));

        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(!NoDataPolicy::is_no_data(i, j)) {
                    typename ArgumentTraits<A1>::value_type a1(fern::get(
                        argument1, i, j));
                    typename ArgumentTraits<A2>::value_type a2(fern::get(
                        argument2, i, j));

                    if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                        NoDataPolicy::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                            result, i, j))) {
                        NoDataPolicy::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        constant_tag,
        array_1d_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A2>::value_type)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + collection
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument2) == fern::size(result));

        size_t const size = fern::size(argument2);

        typename ArgumentTraits<A2>::value_type const& a1(argument1);

        for(size_t i = 0; i < size; ++i) {
            if(!NoDataPolicy::is_no_data(i)) {
                typename ArgumentTraits<A1>::value_type a2(fern::get(
                    argument2, i));

                if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                    NoDataPolicy::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                        result, i))) {
                    NoDataPolicy::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        constant_tag,
        array_2d_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A2>::value_type)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // constant + collection
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument2, 0) == fern::size(result, 0));
        assert(fern::size(argument2, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(argument2, 0);
        size_t const size2 = fern::size(argument2, 1);

        typename ArgumentTraits<A2>::value_type const& a1(argument1);

        for(size_t i = 0; i < size1; ++i) {
            for(size_t j = 0; j < size2; ++j) {
                if(!NoDataPolicy::is_no_data(i, j)) {
                    typename ArgumentTraits<A1>::value_type a2(fern::get(
                        argument2, i, j));

                    if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                        NoDataPolicy::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                            result, i, j))) {
                        NoDataPolicy::mark_as_no_data(i, j);
                    }
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        array_1d_tag,
        constant_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A1>::value_type)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1) == fern::size(result));

        size_t const size = fern::size(argument1);

        typename ArgumentTraits<A2>::value_type const& a2(argument2);

        for(size_t i = 0; i < size; ++i) {
            if(!NoDataPolicy::is_no_data(i)) {
                typename ArgumentTraits<A1>::value_type a1(fern::get(
                    argument1, i));

                if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                    NoDataPolicy::mark_as_no_data(i);
                }

                _algorithm(a1, a2, fern::get(result, i));

                if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                        result, i))) {
                    NoDataPolicy::mark_as_no_data(i);
                }
            }
        }
    }

private:

    Algorithm      _algorithm;

};


template<class A1, class A2, class R,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class Algorithm>
class BinaryOperation<A1, A2, R,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        NoDataPolicy,
        Algorithm,
        array_2d_tag,
        constant_tag>:

    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_arithmetic,
        typename ArgumentTraits<A1>::value_type)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

public:

    BinaryOperation(
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(),
          _algorithm(algorithm)
    {
    }

    BinaryOperation(
        NoDataPolicy&& no_data_policy,
        Algorithm const& algorithm)
        : OutOfDomainPolicy(),
          OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy)),
          _algorithm(algorithm)
    {
    }

    // collection + constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1, 0) == fern::size(result, 0));
        assert(fern::size(argument1, 1) == fern::size(result, 1));

        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, argument1, argument2, result);
    }

    // collection + constant
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(fern::size(argument1, 0) == fern::size(result, 0));
        assert(fern::size(argument1, 1) == fern::size(result, 1));

        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        typename ArgumentTraits<A2>::value_type const& a2(argument2);

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(!NoDataPolicy::is_no_data(i, j)) {
                    typename ArgumentTraits<A1>::value_type a1(fern::get(
                        argument1, i, j));

                    if(!OutOfDomainPolicy::within_domain(a1, a2)) {
                        NoDataPolicy::mark_as_no_data(i, j);
                    }

                    _algorithm(a1, a2, fern::get(result, i, j));

                    if(!OutOfRangePolicy::within_range(a1, a2, fern::get(
                            result, i, j))) {
                        NoDataPolicy::mark_as_no_data(i, j);
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
