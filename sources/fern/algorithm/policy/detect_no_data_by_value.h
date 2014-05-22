#pragma once
#include "fern/core/argument_traits.h"


namespace fern {

//! Input no-data policy class that detect no-data given a special marker value.
/*!
  \tparam    Mask Collection containing the marker values.

  This class keeps a reference to a mask; it doesn't copy the mask. So, copy
  construction and copy assignment are not supported.
*/
template<
    class Mask>
class DetectNoDataByValue {

private:

    using value_type = typename ArgumentTraits<Mask>::value_type;

public:

    bool           is_no_data          () const;

    bool           is_no_data          (size_t index) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2,
                                        size_t index3) const;

                   DetectNoDataByValue (Mask const& mask,
                                        value_type const& no_data_value);

    virtual        ~DetectNoDataByValue()=default;

protected:

                   DetectNoDataByValue ()=delete;

                   DetectNoDataByValue (DetectNoDataByValue const&)=delete;

                   DetectNoDataByValue (DetectNoDataByValue&&)=default;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue const&)=delete;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue&&)=default;

private:

    Mask const&    _mask;

    value_type     _no_data_value;

};


template<
    class Mask>
inline DetectNoDataByValue<Mask>::DetectNoDataByValue(
    Mask const& mask,
    DetectNoDataByValue<Mask>::value_type const& no_data_value)

    : _mask(mask),
      _no_data_value(no_data_value)

{
}


template<
    class Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data() const
{
    return get(_mask) == _no_data_value;
}


template<
    class Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index) const
{
    return get(_mask, index) == _no_data_value;
}


template<
    class Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return get(_mask, index1, index2) == _no_data_value;
}


template<
    class Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index1,
    size_t index2,
    size_t index3) const
{
    return get(_mask, index1, index2, index3) == _no_data_value;
}

} // namespace fern
