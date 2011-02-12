#ifndef INCLUDED_RANALLY_ALGORITHM_PLUS
#define INCLUDED_RANALLY_ALGORITHM_PLUS



namespace ranally {
namespace algorithm {
namespace detail {



} // Namespace detail



template<class Argument1, class Argument2, class Result>
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
  result = argument1 + argument2;
}

} // namespace algorithm
} // namespace ranally

#endif
