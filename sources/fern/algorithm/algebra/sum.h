#pragma once
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

    // constant + constant
    inline void calculate(
        A1 const& argument1,
        R& result)
    {
        result = argument1;
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

    // constant
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

                // TODO
                // if(!OutOfRangePolicy::within_range(a1, result)) {
                //     NoDataPolicy::mark_as_no_data();
                //     break;
                // }
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

    // constant
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

                    // TODO
                    // if(!OutOfRangePolicy::within_range(a1, result)) {
                    //     NoDataPolicy::mark_as_no_data();
                    //     break;
                    // }
                }
            }
        }
    }

private:

};



/// template<class A1, class A2, class R>
/// struct Sum<A1, A2, R,
///     constant_tag,
///     constant_tag>
/// {
/// 
///     // constant, constant
///     inline void calculate(
///         A1 const& argument1,
///         A2 const& argument2,
///         R& result)
///     {
///         result = argument1 == argument2 ? 1 : 0;
///     }
/// 
/// };
/// 
/// 
/// template<class A1, class A2, class R>
/// struct Sum<A1, A2, R,
///     array_1d_tag,
///     constant_tag>
/// {
/// 
///     // collection, constant
///     inline void calculate(
///         A1 const& argument1,
///         A2 const& argument2,
///         R& result)
///     {
///         size_t const size = fern::size(argument1);
///         result = 0;
/// 
///         for(size_t i = 0; i < size; ++i) {
///             if(fern::get(argument1, i) == argument2) {
///                 ++result;
///             }
///         }
///     }
/// 
/// };
/// 
/// 
/// template<class A1, class A2, class R>
/// struct Sum<A1, A2, R,
///     array_2d_tag,
///     constant_tag>
/// {
/// 
///     // collection, constant
///     inline void calculate(
///         A1 const& argument1,
///         A2 const& argument2,
///         R& result)
///     {
///         size_t const size1 = fern::size(argument1, 0);
///         size_t const size2 = fern::size(argument1, 1);
///         result = 0;
/// 
///         for(size_t i = 0; i < size1; ++i) {
///             for(size_t j = 0; j < size2; ++j) {
///                 if(fern::get(argument1, i, j) == argument2) {
///                     ++result;
///                 }
///             }
///         }
///     }
/// 
/// };
/// 
/// 
/// template<class A1, class A2, class R>
/// struct Sum<A1, A2, R,
///     array_3d_tag,
///     constant_tag>
/// {
/// 
///     // collection, constant
///     inline void calculate(
///         A1 const& argument1,
///         A2 const& argument2,
///         R& result)
///     {
///         size_t const size1 = fern::size(argument1, 0);
///         size_t const size2 = fern::size(argument1, 1);
///         size_t const size3 = fern::size(argument1, 2);
///         result = 0;
/// 
///         for(size_t i = 0; i < size1; ++i) {
///             for(size_t j = 0; j < size2; ++j) {
///                 for(size_t k = 0; k < size3; ++k) {
///                     if(fern::get(argument1, i, j, k) == argument2) {
///                         ++result;
///                     }
///                 }
///             }
///         }
///     }
/// 
/// };

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
        // <typename ArgumentTraits<A1>::value_type>,
    class NoDataPolicy=DontMarkNoData
>
struct Sum
{

    //! Type of the result of the operation equals the value type of the argument.
    typedef typename ArgumentTraits<A1>::value_type R;

    typedef typename ArgumentTraits<A1>::value_type A1Value;

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
