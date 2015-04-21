// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>
#include "fern/feature/core/array.h"
#include "fern/feature/core/mask.h"
#include "fern/feature/core/masked_scalar.h"


namespace fern {

//! Class for masked multidimensional arrays.
/*!
    @ingroup    fern_feature_group
    @sa         Array, MaskedRaster

    MaskedArray extents Array with support for masking elements.
*/
template<
    typename T,
    size_t nr_dimensions>
class MaskedArray:
    public Array<T, nr_dimensions>
{

public:

    using Mask = fern::Mask<nr_dimensions>;

                   MaskedArray         ()=default;

                   MaskedArray         (size_t size,
                                        T const& value=T());

                   MaskedArray         (std::initializer_list<T> const& values);

                   MaskedArray         (std::initializer_list<
                                           std::initializer_list<T>> const&
                                              values);

                   MaskedArray         (std::vector<MaskedScalar<T>> const&
                                           values);

    template<size_t nr_ranges>
                   MaskedArray         (gen_type<nr_ranges> const& sizes,
                                        T const& value=T());

                   MaskedArray         (MaskedArray const&)=delete;

    MaskedArray&   operator=           (MaskedArray const&)=delete;

                   MaskedArray         (MaskedArray&&)=default;

    MaskedArray&   operator=           (MaskedArray&&)=delete;

    virtual        ~MaskedArray        ()=default;

    bool           has_masked_values   () const;

    Mask const&    mask                () const;

    Mask&          mask                ();

    template<
        typename U>
    void           mask                (Array<U, nr_dimensions>& mask) const;

    void           set_mask            (T const& value);

    template<
        typename U>
    void           set_mask            (Array<U, nr_dimensions> const& mask,
                                        U const value_to_mask=0);

    void           mask_all            ();

private:

    Mask           _mask;

};


template<
    typename T,
    size_t nr_dimensions>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    std::vector<MaskedScalar<T>> const& values)

    : Array<T, nr_dimensions>(values.size()),
      _mask(values.size())

{
    static_assert(nr_dimensions == 1, "");
    assert(this->num_elements() == values.size());
    assert(_mask.num_elements() == values.size());

    typename Array<T, nr_dimensions>::value_type* value_it = this->data();
    typename Mask::value_type* mask_it = _mask.data();

    for(typename std::vector<MaskedScalar<T>>::const_iterator container_it =
            values.begin(); container_it != values.end();
            ++container_it, ++value_it, ++mask_it) {
        *value_it = (*container_it).value();
        *mask_it = (*container_it).mask();
    }
}


// template<
//     typename T,
//     size_t nr_dimensions>
// template<
//     template<
//         typename>
//     class Container,
//     typename Value>
// inline MaskedArray<T, nr_dimensions>::MaskedArray(
//     Container<Value> const& container)
// 
//     : Array<T, nr_dimensions>(container.size()),
//       _mask(container.size())
// 
// {
//     static_assert(nr_dimensions == 1, "");
//     assert(this->num_elements() == container.size());
//     assert(_mask.num_elements() == container.size());
// 
//     typename Array<T, nr_dimensions>::value_type* value_it = this->data();
//     typename Mask::value_type mask_it = _mask.data();
// 
//     for(typename Container<Value>::const_it container_it = container.begin();
//             container_it != container.end(); ++container_it, ++value_it,
//             ++mask_it) {
//         *value_it = (*container_it).value();
//         *mask_it = (*container_it).mask();
//     }
// }


template<
    typename T,
    size_t nr_dimensions>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    size_t size,
    T const& value)

    : Array<T, nr_dimensions>(size, value),
      _mask(size, false)

{
}


// By default, all values are not masked.
template<
    typename T,
    size_t nr_dimensions>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    std::initializer_list<T> const& values)

    : Array<T, nr_dimensions>(values),
      _mask(extents[values.size()], false)

{
    /// std::fill(_mask.data(), _mask.data() + _mask.num_elements(), false);
}


// By default, all values are not masked.
template<
    typename T,
    size_t nr_dimensions>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    std::initializer_list<std::initializer_list<T>> const& values)

    : Array<T, nr_dimensions>(values),
      _mask(extents[values.size()][values.begin()->size()], false)

{
    /// std::fill(_mask.data(), _mask.data() + _mask.num_elements(), false);
}


// By default, all values are not masked.
template<
    typename T,
    size_t nr_dimensions>
template<
    size_t nr_ranges>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    gen_type<nr_ranges> const& sizes,
    T const& value)

    : Array<T, nr_dimensions>(sizes, value),
      _mask(sizes, false)

{
    /// std::fill(_mask.data(), _mask.data() + _mask.num_elements(), false);
}


template<
    typename T,
    size_t nr_dimensions>
inline bool MaskedArray<T, nr_dimensions>::has_masked_values() const
{
    bool result = false;
    bool const* mask = _mask.data();

    for(size_t i = 0; i < this->num_elements(); ++i) {
        if(mask[i]) {
            result = true;
            break;
        }
    }

    return result;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename MaskedArray<T, nr_dimensions>::Mask const&
    MaskedArray<T, nr_dimensions>::mask() const
{
    return _mask;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename MaskedArray<T, nr_dimensions>::Mask&
    MaskedArray<T, nr_dimensions>::mask()
{
    return _mask;
}


//! Mask all cells that have the same value as \a value passed in.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
template<
    typename T,
    size_t nr_dimensions>
inline void MaskedArray<T, nr_dimensions>::set_mask(
    T const& value)
{
    T* values = this->data();
    bool* mask = _mask.data();

    for(size_t i = 0; i < this->num_elements(); ++i) {
        if(values[i] == value) {
            mask[i] = true;
        }
    }
}


template<
    typename T,
    size_t nr_dimensions>
template<
    typename U>
inline void MaskedArray<T, nr_dimensions>::mask(
    Array<U, nr_dimensions>& mask) const
{
    static_assert(std::is_integral<U>::value, "Mask value must be integral");
    assert(mask.num_elements() == this->num_elements());

    U* other_mask_data = mask.data();
    U const mask_value(0);
    U const non_mask_value(255);
    bool const* mask_data = _mask.data();

    for(size_t i = 0; i < this->num_elements(); ++i) {
        other_mask_data[i] = mask_data[i] ? mask_value : non_mask_value;
    }
}


//! Mask all cells that correspond cells containing \a value_to_mask in the \a mask passed in.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
template<
    typename T,
    size_t nr_dimensions>
template<
    typename U>
inline void MaskedArray<T, nr_dimensions>::set_mask(
    Array<U, nr_dimensions> const& mask,
    U const value_to_mask)
{
    static_assert(std::is_integral<U>::value, "Mask value must be integral");
    assert(mask.num_elements() == this->num_elements());

    U const* other_mask_data = mask.data();
    /// U const mask_value(0);
    bool* mask_data = _mask.data();

    for(size_t i = 0; i < this->num_elements(); ++i) {
        if(other_mask_data[i] == value_to_mask) {
            mask_data[i] = true;
        }
    }
}


template<
    typename T,
    size_t nr_dimensions>
inline void MaskedArray<T, nr_dimensions>::mask_all()
{
    _mask.fill(true);
}

} // namespace fern
