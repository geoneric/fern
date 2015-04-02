#pragma once
#include <cstddef>
#include <utility>
#include "type_traits.h"
#include "customization_point.h"


template<
    typename Mask>
class DetectNoDataByValue
{

private:

    using value_type = ::value_type<Mask>;

public:

    bool           is_no_data          () const;

    bool           is_no_data          (size_t index) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2) const;

                   DetectNoDataByValue (Mask const& mask,
                                        value_type const& no_data_value);

                   DetectNoDataByValue (DetectNoDataByValue const&)=delete;

                   DetectNoDataByValue (DetectNoDataByValue&&)=default;

                   ~DetectNoDataByValue()=default;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue const&)=delete;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue&&)=delete;

private:

    Mask const&    _mask;

    value_type     _no_data_value;

};


template<
    typename Mask>
inline DetectNoDataByValue<Mask>::
        DetectNoDataByValue(
    Mask const& mask,
    value_type const& no_data_value)

    : _mask(mask),
      _no_data_value(no_data_value)

{
}


template<
    typename Mask>
inline bool DetectNoDataByValue<Mask>
        ::is_no_data() const
{
    return get(_mask) == _no_data_value;
}


template<
    typename Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index) const
{
    return get(_mask, index) == _no_data_value;
}


template<
    typename Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return get(_mask, index1, index2) == _no_data_value;
}
