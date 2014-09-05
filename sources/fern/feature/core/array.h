#pragma once
#include <boost/multi_array.hpp>


namespace fern {

extern boost::multi_array_types::extent_gen extents;

extern boost::multi_array_types::index_gen indices;

using Range = boost::multi_array_types::index_range;

template<size_t nr_ranges>
using gen_type = typename boost::detail::multi_array::extent_gen<nr_ranges>;


//! Class for multidimensional arrays.
/*!
    \sa        MaskedArray

    Array is a simple class for managing multidimensional arrays.

    This is how you create a 1-dimensional array:

    \code
    Array<int, 1> array(extents[size]);
    \endcode

    This is how you create a 2-dimensional array:

    \code
    Array<int, 2> array(extents[nr_rows][nr_cols]);
    \endcode

    This class extents boost::multi_array. See the Boost documentation for more
    information.
*/
template<
    class T,
    size_t nr_dimensions>
class Array:
    public boost::multi_array<T, nr_dimensions>
{

public:

                   Array               ()=default;

                   Array               (size_t size,
                                        T const& value=T());

                   Array               (std::initializer_list<T> const& values);

                   Array               (std::initializer_list<
                                           std::initializer_list<T>> const&
                                              values);

                   Array               (std::vector<T> const& values);

    template<size_t nr_ranges>
                   Array               (gen_type<nr_ranges> const& sizes,
                                        T const& value=T());

                   Array               (Array const&)=default;

    Array&         operator=           (Array const&)=default;

                   Array               (Array&&)=default;

    Array&         operator=           (Array&&)=default;

    virtual        ~Array              ()=default;

    void           fill                (T const& value);

private:

};


template<
    class T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::vector<T> const& values)

    : boost::multi_array<T, nr_dimensions>(extents[values.size()])

{
    static_assert(nr_dimensions == 1, "");
    assert(this->num_elements() == values.size());

    std::copy(values.begin(), values.end(), this->data());
}


template<
    class T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    size_t size,
    T const& value)

    : boost::multi_array<T, nr_dimensions>(extents[size])

{
    static_assert(nr_dimensions == 1, "");

    std::fill(this->data(), this->data() + this->num_elements(), value);
}


template<
    class T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline Array<T, nr_dimensions>::Array(
    gen_type<nr_ranges> const& sizes,
    T const& value)

    : boost::multi_array<T, nr_dimensions>(sizes)

{
    std::fill(this->data(), this->data() + this->num_elements(), value);
}


template<
    class T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::initializer_list<T> const& values)

    : boost::multi_array<T, nr_dimensions>(
          extents[values.size()])

{
    static_assert(nr_dimensions == 1, "");

    T* it = this->data();

    for(auto const& value: values) {
        *it++ = value;
    }
}


template<
    class T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::initializer_list<std::initializer_list<T>> const& values)

    : boost::multi_array<T, nr_dimensions>(
          extents[values.size()][values.begin()->size()])

{
    static_assert(nr_dimensions == 2, "");

    T* it = this->data();

    for(auto const& row: values) {
        for(auto const& col: row) {
            *it++ = col;
        }
    }
}


template<
    class T,
    size_t nr_dimensions>
inline void Array<T, nr_dimensions>::fill(
    T const& value)
{
    std::fill(this->data(), this->data() + this->num_elements(), value);
}

} // namespace fern
