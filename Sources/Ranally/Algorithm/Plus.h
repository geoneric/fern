#ifndef INCLUDED_RANALLY_ALGORITHM_PLUS
#define INCLUDED_RANALLY_ALGORITHM_PLUS

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/const_iterator.hpp>



namespace ranally {
namespace algorithm {
namespace detail {

// template<
//   typename Raster>
// struct RasterTraits
// {
//   static bool const isConstant = false;
// };
// 
// template<>
// struct RasterTraits<int>
// {
//   static bool const isConstant = true;
// };



template<
  typename Argument>
struct ArgumentTraits
{
  static bool const isScalar=false;
};



template<>
struct ArgumentTraits<int>
{
  static bool const isScalar=true;
};



bool const IS_SCALAR = true;
bool const IS_NOT_SCALAR = false;

template<
  typename Argument1,
  typename Argument2,
  typename Result,
  bool argument1IsScalar,
  bool argument2IsScalar>
struct Plus
{
};



template<
  typename Argument1,
  typename Argument2,
  typename Result>
struct Plus<Argument1, Argument2, Result, IS_SCALAR, IS_SCALAR>
{
  static void calculate(
    Argument1 const& argument1,
    Argument2 const& argument2,
    Result& result)
  {
    result = argument1 + argument2;
  }
};



template<
  typename Argument1,
  typename Argument2,
  typename Result>
struct Plus<Argument1, Argument2, Result, IS_SCALAR, IS_NOT_SCALAR>
{
  static void calculate(
    Argument1 const& argument1,
    Argument2 const& argument2,
    Result& result)
  {
    // typedef typename 
    typename boost::range_const_iterator<Argument2>::type argument2It =
      boost::begin(argument2);
    typename boost::range_const_iterator<Argument2>::type const end2 =
      boost::end(argument2);
    typename boost::range_iterator<Result>::type resultIt =
      boost::begin(result);

    for(; argument2It != end2; ++argument2It, ++resultIt) {
      *resultIt = argument1 + *argument2It;
    }
  }
};



template<
  typename Argument1,
  typename Argument2,
  typename Result>
struct Plus<Argument1, Argument2, Result, IS_NOT_SCALAR, IS_SCALAR>
{
  static void calculate(
    Argument1 const& argument1,
    Argument2 const& argument2,
    Result& result)
  {
    // typedef typename 
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
};




template<
  typename Argument1,
  typename Argument2,
  typename Result>
struct Plus<Argument1, Argument2, Result, IS_NOT_SCALAR, IS_NOT_SCALAR>
{
  static void calculate(
    Argument1 const& argument1,
    Argument2 const& argument2,
    Result& result)
  {
    // typedef typename 
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
};

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
  // Switch on argument type:
  // Scalars
  //   Use interface of InputScalar concept.
  // Rasters
  //   Use interface of InputRaster concept.
  // Order the arguments: Scalar, Raster, to limit the amount of overloads to
  // create.

  detail::Plus<Argument1, Argument2, Result,
    detail::ArgumentTraits<Argument1>::isScalar,
    detail::ArgumentTraits<Argument2>::isScalar>::calculate(argument1,
      argument2, result);

  // Iteration must be handled per argument.
  // We must end up with pieces of code that contain the loop. At that level
  // we can decide how to iterate (1D array, 2D array, vector, Raster, ...).
  // Treat scalars and arrays special. The rest must adhere to the Scalar and
  // Raster concept.
}

} // namespace algorithm
} // namespace ranally

#endif
