#pragma once
#include "fern/core/assert.h"
#include "fern/core/collection_traits.h"
#include "fern/core/constant_traits.h"
#include "fern/feature/core/masked_constant.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"


namespace fern {
namespace sum {
namespace detail {
namespace dispatch {

template<class A1, class R,
    class OutOfRangePolicy,
    class NoDataPolicy,
    class A1CollectionCategory>
struct Sum
{
};


template<class A1, class R,
    class OutOfRangePolicy,
    class NoDataPolicy>
class Sum<A1, R,
        OutOfRangePolicy,
        NoDataPolicy,
        constant_tag>:

    public OutOfRangePolicy,
    public NoDataPolicy

{

public:

    Sum()
        : OutOfRangePolicy(),
          NoDataPolicy()
    {
    }

    Sum(
        NoDataPolicy&& no_data_policy)
        : OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy))
    {
    }

    // constant
    inline void calculate(
        A1 const& argument1,
        R& result)
    {
        if(!NoDataPolicy::is_no_data()) {
            typename ArgumentTraits<A1>::value_type a1(fern::get(argument1));
            fern::get(result) = fern::get(argument1);

            if(!OutOfRangePolicy::within_range(a1, result)) {
                NoDataPolicy::mark_as_no_data();
            }
        }
    }

private:

};


template<class A1, class R,
    class OutOfRangePolicy,
    class NoDataPolicy>
class Sum<A1, R,
        OutOfRangePolicy,
        NoDataPolicy,
        array_1d_tag>:

    public OutOfRangePolicy,
    public NoDataPolicy

{

public:

    Sum()
        : OutOfRangePolicy(),
          NoDataPolicy()
    {
    }

    Sum(
        NoDataPolicy&& no_data_policy)
        : OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy))
    {
    }

    // 1d array
    inline void calculate(
        A1 const& argument1,
        R& result)
    {
        size_t const size = fern::size(argument1);
        result = 0;

        for(size_t i = 0; i < size; ++i) {
            if(!NoDataPolicy::is_no_data(i)) {
                typename ArgumentTraits<A1>::value_type a1(fern::get(
                    argument1, i));

                result += a1;

                if(!OutOfRangePolicy::within_range(a1, result)) {
                    NoDataPolicy::mark_as_no_data();
                    break;
                }
            }
        }
    }

private:

};


template<class A1, class R,
    class OutOfRangePolicy,
    class NoDataPolicy>
class Sum<A1, R,
        OutOfRangePolicy,
        NoDataPolicy,
        array_2d_tag>:

    public OutOfRangePolicy,
    public NoDataPolicy

{

public:

    Sum()
        : OutOfRangePolicy(),
          NoDataPolicy()
    {
    }

    Sum(
        NoDataPolicy&& no_data_policy)
        : OutOfRangePolicy(),
          NoDataPolicy(std::forward<NoDataPolicy>(no_data_policy))
    {
    }

    // 2d array
    inline void calculate(
        A1 const& argument1,
        R& result)
    {
        size_t const size1 = fern::size(argument1, 0);
        size_t const size2 = fern::size(argument1, 1);
        result = 0;

        for(size_t i = 0; i < size1; ++i) {
            for(size_t j = 0; j < size2; ++j) {
                if(!NoDataPolicy::is_no_data(i, j)) {
                    typename ArgumentTraits<A1>::value_type a1(fern::get(
                        argument1, i, j));

                    result += a1;

                    if(!OutOfRangePolicy::within_range(a1, result)) {
                        NoDataPolicy::mark_as_no_data();
                        break;
                    }
                }
            }
        }
    }

private:

};

} // namespace dispatch
} // namespace detail
} // namespace sum


namespace algebra {

//! Implementation of the sum operation.
/*!
  \tparam    A1 Type of first argument.
*/
template<
    class A1,
    class OutOfRangePolicy=DiscardRangeErrors,
    class NoDataPolicy=DontMarkNoData
>
struct Sum
{

    typedef typename ArgumentTraits<A1>::value_type A1Value;

    FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)

    // We see boolean more as nominal values than as integers. You can't sum'em.
    FERN_STATIC_ASSERT(!std::is_same, A1Value, bool)

    //! Type of the result of the operation equals the value type of the argument.
    // typedef typename ArgumentTraits<A1>::value_type R;
    typedef typename boost::mpl::if_<
        boost::mpl::bool_<ArgumentTraits<A1>::is_masking>,
        MaskedConstant<A1Value>,
        A1Value>::type R;

    Sum()
        : algorithm()
    {
    }

    Sum(
        NoDataPolicy&& no_data_policy)
        : algorithm(std::forward<NoDataPolicy>(no_data_policy))
    {
    }

    inline void operator()(
        A1 const& argument1,
        R& result)
    {
        algorithm.calculate(argument1, result);
    }

    sum::detail::dispatch::Sum<A1, R,
        OutOfRangePolicy, NoDataPolicy,
        typename ArgumentTraits<A1>::argument_category> algorithm;

};


template<
    class A1>
void sum(
    A1 const& argument1,
    typename Sum<A1>::R& result)
{
    Sum<A1>()(argument1, result);
}

} // namespace algebra
} // namespace fern
