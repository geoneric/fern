#pragma once
#include "Ranally/data_traits.h"


namespace ranally {
namespace algorithm {


template<
    typename Argument1,
    typename Argument2,
    typename Result>
inline void plus(
    Argument1 const& argument1,
    Argument2 const& argument2,
    Result& result)
{
    typedef typename DataTraits<Argument1>::DataCategory category1;
    typedef typename DataTraits<Argument2>::DataCategory category2;
    // detail::average(argument1, category1(), argument2, category2(), result);
}

} // namespace algorithm
} // namespace ranally
