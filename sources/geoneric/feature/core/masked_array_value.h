#pragma once
#include <geoneric/feature/core/masked_array.h>
#include <geoneric/feature/core/value.h>


namespace geoneric {

//! Multi-dimensional array value.
/*!
  This class represents an n-dimensional array.

  \sa        .
*/
template<
    class T,
    size_t nr_dimensions>
class MaskedArrayValue:
    public MaskedArray<T, nr_dimensions>,
    public Value
{

public:

                   MaskedArrayValue    ()=default;

    template<class ExtentList>
                   MaskedArrayValue    (ExtentList const& sizes);

                   MaskedArrayValue    (MaskedArrayValue const&)=delete;

    MaskedArrayValue& operator=        (MaskedArrayValue const&)=delete;

                   MaskedArrayValue    (MaskedArrayValue&&)=delete;

    MaskedArrayValue& operator=        (MaskedArrayValue&&)=delete;

                   ~MaskedArrayValue   ()=default;

private:

};


template<
    class T,
    size_t nr_dimensions>
template<
    class ExtentList>
inline MaskedArrayValue<T, nr_dimensions>::MaskedArrayValue(
    ExtentList const& sizes)

    : MaskedArray<T, nr_dimensions>(sizes),
      Value()

{
}

} // namespace geoneric
