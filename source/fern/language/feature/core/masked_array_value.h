// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/masked_array.h"
#include "fern/language/feature/core/value.h"


namespace fern {
namespace language {

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

} // namespace language
} // namespace fern
