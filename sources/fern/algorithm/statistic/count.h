#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace count {
namespace detail {
namespace dispatch {

template<class A1, class A2, class R,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct Count
{
};


template<class A1, class A2, class R>
struct Count<A1, A2, R,
    constant_tag,
    constant_tag>
{

    // constant, constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        result = argument1 == argument2 ? 1 : 0;
    }

};


template<class A1, class A2, class R>
struct Count<A1, A2, R,
    array_1d_tag,
    constant_tag>
{

    // collection, constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        size_t const size = fern::size(argument1);
        result = 0;

        for(size_t i = 0; i < size; ++i) {
            if(fern::get(argument1, i) == argument2) {
                ++result;
            }
        }
    }

};


template<class A1, class A2, class R>
struct Count<A1, A2, R,
    array_2d_tag,
    constant_tag>
{

    // collection, constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {

        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, argument1, argument2, result);
    }

    // collection, constant
    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        size_t const start1 = indices[0].begin();
        size_t const finish1 = indices[0].end();
        size_t const start2 = indices[1].begin();
        size_t const finish2 = indices[1].end();

        result = 0;

        for(size_t i = start1; i < finish1; ++i) {
            for(size_t j = start2; j < finish2; ++j) {
                if(fern::get(argument1, i, j) == argument2) {
                    ++result;
                }
            }
        }
    }

};


template<class A1, class A2, class R>
struct Count<A1, A2, R,
    array_3d_tag,
    constant_tag>
{

    // collection, constant
    inline void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        size_t const size3 = fern::size(argument1, 2);
        result = 0;

        for(size_t i = 0; i < size1; ++i) {
            for(size_t j = 0; j < size2; ++j) {
                for(size_t k = 0; k < size3; ++k) {
                    if(fern::get(argument1, i, j, k) == argument2) {
                        ++result;
                    }
                }
            }
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace count


namespace algebra {

//! Implementation of the count operation.
/*!
  \tparam    A1 Type of first argument.
  \tparam    A2 Type of second argument.

  Check for out of range results is not needed, assuming the result type is
  capable of storing at least the number of values in the input.
*/
template<
    class A1,
    class A2
>
struct Count
{

    typedef local_aggregate_operation_tag category;

    //! Type of the result of the operation.
    typedef size_t R;

    typedef typename ArgumentTraits<A1>::value_type A1Value;

    typedef typename ArgumentTraits<A2>::value_type A2Value;

    Count()
        : algorithm()
    {
    }

    inline void operator()(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        algorithm.calculate(argument1, argument2, result);
    }

    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        algorithm.calculate(indices, argument1, argument2, result);
    }

    template<
        class Collection>
    inline static void aggregate(
        Collection const& results,
        R& result)
    {
        FERN_STATIC_ASSERT(std::is_arithmetic, R);
        FERN_STATIC_ASSERT(std::is_same,
            typename ArgumentTraits<Collection>::value_type, R);
        sum(results, result);
    }

    count::detail::dispatch::Count<A1, A2, R,
        typename ArgumentTraits<A1>::argument_category,
        typename ArgumentTraits<A2>::argument_category> algorithm;

};


template<
    class A1,
    class A2>
void count(
    A1 const& argument1,
    A2 const& argument2,
    typename Count<A1, A2>::R& result)
{
    Count<A1, A2>()(argument1, argument2, result);
}

} // namespace algebra
} // namespace fern
