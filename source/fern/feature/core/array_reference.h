#pragma once
#include <boost/multi_array.hpp>


namespace fern {

extern boost::multi_array_types::extent_gen extents;

extern boost::multi_array_types::index_gen indices;

using Range = boost::multi_array_types::index_range;

template<size_t nr_ranges>
using gen_type = typename boost::detail::multi_array::extent_gen<nr_ranges>;


template<
    class T,
    size_t nr_dimensions>
class ArrayReference:
    public boost::multi_array_ref<T, nr_dimensions /* , T* */>
{

public:

    template<
        size_t nr_ranges>
                   ArrayReference      (T* values,
                                        gen_type<nr_ranges> const& sizes);

private:

};


template<
    class T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline ArrayReference<T, nr_dimensions>::ArrayReference(
    T* values,
    gen_type<nr_ranges> const& sizes)

    : boost::multi_array_ref<T, nr_dimensions /* , T const* */>(values, sizes)

{
}

} // namespace fern
