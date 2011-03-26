#ifndef INCLUDED_RANALLY_ALGORITHM_PLUS
#define INCLUDED_RANALLY_ALGORITHM_PLUS

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/const_iterator.hpp>
#include "Ranally/DataTraits.h"



namespace ranally {
namespace algorithm {
namespace detail {

template<
  typename Argument1,
  typename Argument2,
  typename Result>
inline void plus(
  Argument1 const& argument1,
  IterableTag /* tag1 */,
  Argument2 const& argument2,
  ScalarTag /* tag2 */,
  Result& result)
{
  typename boost::range_const_iterator<Argument1>::type argument1It =
    boost::begin(argument1);
  typename boost::range_const_iterator<Argument1>::type const end1 =
    boost::end(argument1);
  typename boost::range_iterator<Result>::type resultIt =
    boost::begin(result);

  for(; argument1It != end1; ++argument1It, ++resultIt) {
    *resultIt = *argument1It + argument2;
  }
}

template<
  typename Argument1,
  typename Argument2,
  typename Result>
inline void plus(
  Argument1 argument1,
  ScalarTag tag1,
  Argument2 const& argument2,
  IterableTag tag2,
  Result& result)
{
  // Reorder the arguments and call the other algorithm.
  plus(argument2, tag2, argument1, tag1, result);
}

template<
  typename Argument1,
  typename Argument2,
  typename Result>
inline void plus(
  Argument1 argument1,
  ScalarTag /* tag1 */,
  Argument2 argument2,
  ScalarTag /* tag2 */,
  Result& result)
{
  result = argument1 + argument2;
}

template<
  typename Argument1,
  typename Argument2,
  typename Result>
inline void plus(
  Argument1 const& argument1,
  IterableTag /* tag1 */,
  Argument2 const& argument2,
  IterableTag /* tag2 */,
  Result& result)
{
  typename boost::range_const_iterator<Argument1>::type argument1It =
    boost::begin(argument1);
  typename boost::range_const_iterator<Argument2>::type argument2It =
    boost::begin(argument2);
  typename boost::range_const_iterator<Argument1>::type const end1 =
    boost::end(argument1);
  typename boost::range_iterator<Result>::type resultIt =
    boost::begin(result);

  for(; argument1It != end1; ++argument1It, ++argument2It, ++resultIt) {
    *resultIt = *argument1It + *argument2It;
  }
}

} // Namespace detail



//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
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
  detail::plus(argument1, category1(), argument2, category2(), result);
}

} // namespace algorithm
} // namespace ranally

#endif
