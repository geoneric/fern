#ifndef INCLUDED_RANALLY_OPERATION_PRINT
#define INCLUDED_RANALLY_OPERATION_PRINT

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/distance.hpp>
#include "Ranally/DataTraits.h"



namespace ranally {
namespace operation {
namespace detail {

template<
  typename Argument>
inline void print(
  ScalarTag /* tag */,
  Argument argument,
  std::ostream& stream)
{
  stream << argument << '\n';
}



template<
  typename Argument>
inline void print(
  RangeTag /* tag */,
  Argument const& argument,
  std::ostream& stream)
{
  stream << '[';

  size_t distance = boost::distance(argument);

  if(distance > 0) {
    typename boost::range_iterator<Argument const>::type pos =
      boost::const_begin(argument);
    typename boost::range_iterator<Argument const>::type end =
      boost::const_end(argument);

    if(distance < 7) {
      stream << *pos++;
      while(pos != end) {
        stream << ", " << *pos++;
      }
    }
    else {
      stream << *pos++ << ", ";
      stream << *pos++ << ", ";
      stream << *pos++ << ", ..., ";
      stream << *(end - 3) << ", " << *(end - 2) << ", " << *(end - 1);
    }
  }

  stream << "]\n";
}



template<
  typename Argument>
inline void print(
  RasterTag /* tag */,
  Argument const& /* argument */,
  std::ostream& stream)
{
  stream << '[';

  stream << "]\n";
}

} // namespace detail



template<
  typename Argument>
inline void print(
  Argument const& argument,
  std::ostream& stream)
{
  typedef typename DataTraits<Argument>::DataCategory category;
  detail::print(category(), argument, stream);
}

} // namespace operation
} // namespace ranally

#endif
