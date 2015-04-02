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
    @ingroup    fern_feature_group
    @sa         MaskedArray

    Array is a simple class for managing multidimensional arrays.

    This is how you create a 1-dimensional array:

    @code
    Array<int, 1> array(extents[size]);
    @endcode

    This is how you create a 2-dimensional array:

    @code
    Array<int, 2> array(extents[nr_rows][nr_cols]);
    @endcode

    For these two cases, shortcuts exist:

    @code
    Array<int, 1> array1d(size);
    Array<int, 2> array2d(size1, size2);
    @endcode
*/
template<
    typename T,
    size_t nr_dimensions>
class Array
{

public:

    using value_type = typename
        boost::multi_array<T, nr_dimensions>::value_type;

    using reference = typename
        boost::multi_array<T, nr_dimensions>::reference;

    using const_reference = typename
        boost::multi_array<T, nr_dimensions>::const_reference;

    using size_type = typename
        boost::multi_array<T, nr_dimensions>::size_type;

    using iterator = typename
        boost::multi_array<T, nr_dimensions>::iterator;

    using const_iterator = typename
        boost::multi_array<T, nr_dimensions>::const_iterator;

                   Array               ()=default;

                   Array               (size_t size,
                                        T const& value=T());

                   Array               (size_t size1,
                                        size_t size2,
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

    size_t         size                () const;

    size_t         num_elements        () const;

    size_t         num_dimensions      () const;

    size_type const*
                   shape               () const;

    reference      operator[]          (size_t index);

    const_reference
                   operator[]          (size_t index) const;

    iterator       begin               ();

    iterator       end                 ();

    const_iterator begin               () const;

    const_iterator end                 () const;

    T*             data                ();

    T const*       data                () const;

    void           fill                (T const& value);

private:

    boost::multi_array<T, nr_dimensions> _array;

};


template<
    typename T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::vector<T> const& values)

    : _array(extents[values.size()])

{
    static_assert(nr_dimensions == 1, "");
    assert(this->num_elements() == values.size());

    std::copy(values.begin(), values.end(), this->data());
}


template<
    typename T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    size_t size,
    T const& value)

    : _array(extents[size])

{
    static_assert(nr_dimensions == 1, "");

    std::fill(this->data(), this->data() + this->num_elements(), value);
}


template<
    typename T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    size_t size1,
    size_t size2,
    T const& value)

    : _array(extents[size1][size2])

{
    static_assert(nr_dimensions == 2, "");

    std::fill(this->data(), this->data() + this->num_elements(), value);
}


template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline Array<T, nr_dimensions>::Array(
    gen_type<nr_ranges> const& sizes,
    T const& value)

    : _array(sizes)

{
    std::fill(this->data(), this->data() + this->num_elements(), value);
}


template<
    typename T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::initializer_list<T> const& values)

    : _array(extents[values.size()])

{
    static_assert(nr_dimensions == 1, "");

    T* it = this->data();

    for(auto const& value: values) {
        *it++ = value;
    }
}


template<
    typename T,
    size_t nr_dimensions>
inline Array<T, nr_dimensions>::Array(
    std::initializer_list<std::initializer_list<T>> const& values)

    : _array(extents[values.size()][values.begin()->size()])

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
    typename T,
    size_t nr_dimensions>
inline size_t Array<T, nr_dimensions>::size() const
{
    return _array.size();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t Array<T, nr_dimensions>::num_elements() const
{
    return _array.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t Array<T, nr_dimensions>::num_dimensions() const
{
    return _array.num_dimensions();
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::size_type const*
         Array<T, nr_dimensions>::shape() const
{
    return _array.shape();
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::reference
        Array<T, nr_dimensions>::operator[](
    size_t index)
{
    return _array[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::const_reference
        Array<T, nr_dimensions>::operator[](
    size_t index) const
{
    return _array[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::iterator
        Array<T, nr_dimensions>::begin()
{
    return _array.begin();
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::iterator
        Array<T, nr_dimensions>::end()
{
    return _array.end();
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::const_iterator
        Array<T, nr_dimensions>::begin() const
{
    return _array.begin();
}


template<
    typename T,
    size_t nr_dimensions>
inline typename Array<T, nr_dimensions>::const_iterator
        Array<T, nr_dimensions>::end() const
{
    return _array.end();
}


template<
    typename T,
    size_t nr_dimensions>
inline T* Array<T, nr_dimensions>::data()
{
    return _array.data();
}


template<
    typename T,
    size_t nr_dimensions>
inline T const* Array<T, nr_dimensions>::data() const
{
    return _array.data();
}


template<
    typename T,
    size_t nr_dimensions>
inline void Array<T, nr_dimensions>::fill(
    T const& value)
{
    std::fill(this->data(), this->data() + this->num_elements(), value);
}

} // namespace fern
