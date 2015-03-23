#pragma once
#include "fern/feature/core/array.h"
#include "fern/feature/core/value.h"


namespace fern {

//! Multi-dimensional array value.
/*!
    @brief      This class represents an n-dimensional array.
*/
template<
    typename T,
    size_t nr_dimensions>
class ArrayValue:
    public Array<T, nr_dimensions>,
    public Value
{

public:

                   ArrayValue          ()=default;

    template<size_t nr_ranges>
                   ArrayValue          (gen_type<nr_ranges> const& sizes);

                   ArrayValue          (ArrayValue const&)=delete;

    ArrayValue&    operator=           (ArrayValue const&)=delete;

                   ArrayValue          (ArrayValue&&)=delete;

    ArrayValue&    operator=           (ArrayValue&&)=delete;

                   ~ArrayValue         ()=default;

private:

};


template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline ArrayValue<T, nr_dimensions>::ArrayValue(
    gen_type<nr_ranges> const& sizes)

    : Array<T, nr_dimensions>(sizes),
      Value()

{
}

} // namespace fern
