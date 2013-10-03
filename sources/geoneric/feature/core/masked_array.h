#pragma once
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

    Array<bool, nr_dimensions> const& mask() const;

    Array<bool, nr_dimensions>& mask   ();

    void           mask_value          (T const& value);

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


template<
    class T,
    size_t nr_dimensions>
inline void MaskedArray<T, nr_dimensions>::mask_value(
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

} // namespace geoneric
