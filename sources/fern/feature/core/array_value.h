#pragma once
#include "fern/feature/core/array.h"
#include "fern/feature/core/value.h"


namespace fern {

//! Multi-dimensional array value.
/*!
  This class represents an n-dimensional array.

  \sa        .
*/
template<
    class T,
    size_t nr_dimensions>
class ArrayValue:
    public Array<T, nr_dimensions>,
    public Value
{

public:

                   ArrayValue          ()=default;

    template<class ExtentList>
                   ArrayValue          (ExtentList const& sizes);

                   ArrayValue          (ArrayValue const&)=delete;

    ArrayValue&    operator=           (ArrayValue const&)=delete;

                   ArrayValue          (ArrayValue&&)=delete;

    ArrayValue&    operator=           (ArrayValue&&)=delete;

                   ~ArrayValue         ()=default;

private:

};


template<
    class T,
    size_t nr_dimensions>
template<
    class ExtentList>
inline ArrayValue<T, nr_dimensions>::ArrayValue(
    ExtentList const& sizes)

    : Array<T, nr_dimensions>(sizes),
      Value()

{
}

} // namespace fern
