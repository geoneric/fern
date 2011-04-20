#ifndef INCLUDED_RANALLY_OPERATION_PRINT
#define INCLUDED_RANALLY_OPERATION_PRINT

#include "Ranally/DataTraits.h"



namespace ranally {
namespace operation {
namespace detail {

// template<
//   typename DataCategory,
//   typename Argument>
// void               print               (Argument const& argument,
//                                         std::ostream& stream);



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
  Argument const& /* argument */,
  std::ostream& stream)
{
  stream << '[';

  // size_t distance = boost::distance(argument);

  // << *argument

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
