#pragma once
#include <cstddef>
#include "type_traits.h"
#include "customization_point.h"


template<
    typename Mask>
class MarkNoDataByValue {

private:

    using value_type = typename TypeTraits<Mask>::value_type;

public:

    void           mark_as_no_data     ();

    void           mark_as_no_data     (size_t index);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2);

                   MarkNoDataByValue   (Mask& mask,
                                        value_type const& no_data_value);

                   MarkNoDataByValue   (MarkNoDataByValue&&)=delete;

                   MarkNoDataByValue   (MarkNoDataByValue const&)=delete;

                   ~MarkNoDataByValue  ()=default;

    MarkNoDataByValue
                   operator=           (MarkNoDataByValue&&)=delete;

    MarkNoDataByValue
                   operator=           (MarkNoDataByValue const&)=delete;

private:

    Mask&          _mask;

    value_type     _no_data_value;

};


template<
    typename Mask>
inline MarkNoDataByValue<Mask>::MarkNoDataByValue(
    Mask& mask,
    MarkNoDataByValue<Mask>::value_type const& no_data_value)

    : _mask(mask),
      _no_data_value(no_data_value)

{
}


template<
    typename Mask>
inline void MarkNoDataByValue<Mask>::mark_as_no_data()
{
    get(_mask) = _no_data_value;
}


template<
    typename Mask>
inline void MarkNoDataByValue<Mask>::mark_as_no_data(
    size_t index)
{
    get(_mask, index) = _no_data_value;
}


template<
    typename Mask>
inline void MarkNoDataByValue<Mask>::mark_as_no_data(
    size_t index1,
    size_t index2)
{
    get(_mask, index1, index2) = _no_data_value;
}
