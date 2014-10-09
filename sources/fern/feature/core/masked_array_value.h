#pragma once
#include "fern/feature/core/masked_array.h"
#include "fern/feature/core/value.h"


namespace fern {

//! Multi-dimensional array value.
/*!
  This class represents an n-dimensional array.

  \sa        .
*/
template<
    typename T,
    size_t nr_dimensions>
class MaskedArrayValue:
    public MaskedArray<T, nr_dimensions>,
    public Value
{

public:

                   MaskedArrayValue    ()=default;

    template<size_t nr_ranges>
                   MaskedArrayValue    (gen_type<nr_ranges> const& sizes);

                   MaskedArrayValue    (MaskedArrayValue const&)=delete;

    MaskedArrayValue& operator=        (MaskedArrayValue const&)=delete;

                   MaskedArrayValue    (MaskedArrayValue&&)=delete;

    MaskedArrayValue& operator=        (MaskedArrayValue&&)=delete;

                   ~MaskedArrayValue   ()=default;

private:

};


template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline MaskedArrayValue<T, nr_dimensions>::MaskedArrayValue(
    gen_type<nr_ranges> const& sizes)

    : MaskedArray<T, nr_dimensions>(sizes),
      Value()

{
}

} // namespace fern
