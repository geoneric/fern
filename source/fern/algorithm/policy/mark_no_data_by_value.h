// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include "fern/core/data_type_traits.h"
#include "fern/algorithm/core/mask_customization_point.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Output no-data policy class that marks no-data given a
                special marker value.
    @tparam     Mask Collection receiving the marker values.

    This class keeps a reference to a mask; it doesn't copy the mask. So,
    copy construction and copy assignment are not supported.
*/
template<
    typename Mask>
class MarkNoDataByValue {

private:

    using value_type = typename DataTypeTraits<Mask>::value_type;

public:

    void           mark_as_no_data     ();

    void           mark_as_no_data     (size_t index);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2);

    void           mark_as_no_data     (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   MarkNoDataByValue   (Mask& mask);

                   MarkNoDataByValue   (Mask& mask,
                                        value_type const& no_data_value);

                   ~MarkNoDataByValue  ()=default;

protected:

                   MarkNoDataByValue   ()=delete;

                   MarkNoDataByValue   (MarkNoDataByValue const&)=delete;

                   MarkNoDataByValue   (MarkNoDataByValue&&)=default;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue const&)=delete;

    MarkNoDataByValue&
                   operator=           (MarkNoDataByValue&&)=default;

private:

    Mask&          _mask;

    value_type     _no_data_value;

};


template<
    typename Mask>
inline MarkNoDataByValue<Mask>::MarkNoDataByValue(
    Mask& mask)

    : _mask(mask),
      _no_data_value(no_data_value(_mask))

{
}


template<
    typename Mask>
inline MarkNoDataByValue<Mask>::MarkNoDataByValue(
    Mask& mask,
    typename MarkNoDataByValue<Mask>::value_type const& no_data_value)

    : _mask(mask),
      _no_data_value(no_data_value)

{
}


template<
    typename Mask>
inline void MarkNoDataByValue<Mask>::mark_as_no_data()
{
    // In case of a compile error, make sure that get is overloaded for
    // Mask. This is not the case for regular constants. You may need to
    // pick a type like MaskedConstant, which supports masking.
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


template<
    typename Mask>
inline void MarkNoDataByValue<Mask>::mark_as_no_data(
    size_t index1,
    size_t index2,
    size_t index3)
{
    get(_mask, index1, index2, index3) = _no_data_value;
}

} // namespace algorithm
} // namespace fern
