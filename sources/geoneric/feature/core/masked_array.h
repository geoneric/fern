#pragma once
#include <type_traits>
#include <geoneric/feature/core/array.h>


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class T,
    size_t nr_dimensions>
class MaskedArray:
    public Array<T, nr_dimensions>
{

public:

                   MaskedArray         ()=default;

    template<class ExtentList>
                   MaskedArray         (ExtentList const& sizes);

                   MaskedArray         (MaskedArray const&)=delete;

    MaskedArray&   operator=           (MaskedArray const&)=delete;

                   MaskedArray         (MaskedArray&&)=delete;

    MaskedArray&   operator=           (MaskedArray&&)=delete;

    virtual        ~MaskedArray        ();

    bool           has_masked_values   () const;

    Array<bool, nr_dimensions> const& mask() const;

    Array<bool, nr_dimensions>& mask   ();

    template<
        class U>
    void           mask                (Array<U, nr_dimensions>& mask) const;

    void           set_mask            (T const& value);

    template<
        class U>
    void           set_mask            (Array<U, nr_dimensions> const& mask);

private:

    Array<bool, nr_dimensions> _mask;

};


template<
    class T,
    size_t nr_dimensions>
template<
    class ExtentList>
inline MaskedArray<T, nr_dimensions>::MaskedArray(
    ExtentList const& sizes)

    : Array<T, nr_dimensions>(sizes),
      _mask(sizes)

{
    // By default, all values are not masked.
    std::fill(_mask.data(), _mask.data() + _mask.size(), false);
}


template<
    class T,
    size_t nr_dimensions>
inline MaskedArray<T, nr_dimensions>::~MaskedArray()
{
}


template<
    class T,
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
    class T,
    size_t nr_dimensions>
inline Array<bool, nr_dimensions> const& MaskedArray<T, nr_dimensions>::
    mask() const
{
    return _mask;
}


template<
    class T,
    size_t nr_dimensions>
inline Array<bool, nr_dimensions>& MaskedArray<T, nr_dimensions>::mask()
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
    class T,
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
    class T,
    size_t nr_dimensions>
template<
    class U>
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


//! Mask all cells that correspond cells containing zero in the \a mask passed in.
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
template<
    class T,
    size_t nr_dimensions>
template<
    class U>
inline void MaskedArray<T, nr_dimensions>::set_mask(
    Array<U, nr_dimensions> const& mask)
{
    static_assert(std::is_integral<U>::value, "Mask value must be integral");
    assert(mask.num_elements() == this->num_elements());

    U const* other_mask_data = mask.data();
    U const mask_value(0);
    bool* mask_data = _mask.data();

    for(size_t i = 0; i < this->num_elements(); ++i) {
        if(other_mask_data[i] == mask_value) {
            mask_data[i] = true;
        }
    }
}

} // namespace geoneric
