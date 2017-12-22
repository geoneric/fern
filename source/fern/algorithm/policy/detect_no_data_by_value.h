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
#include <utility>
#include "fern/core/data_type_traits.h"
#include "fern/core/math.h"
#include "fern/algorithm/core/mask_customization_point.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Input no-data policy class that detect no-data given a
                special marker value.
    @tparam     Mask Collection containing the marker values.

    This class keeps a reference to a mask; it doesn't copy the mask. So,
    copy construction and copy assignment are not supported.
*/
template<
    typename Mask>
class DetectNoDataByValue
{

private:

    using value_type = fern::value_type<Mask>;

public:

    bool           is_no_data          () const;

    bool           is_no_data          (size_t index) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2) const;

    bool           is_no_data          (size_t index1,
                                        size_t index2,
                                        size_t index3) const;

                   DetectNoDataByValue (DetectNoDataByValue const& other)
                                            =default;

                   DetectNoDataByValue (DetectNoDataByValue&& other) =default;

                   DetectNoDataByValue (Mask const& mask);

                   DetectNoDataByValue (Mask const& mask,
                                        value_type const& no_data_value);

                   ~DetectNoDataByValue()=default;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue const&)=default;

    DetectNoDataByValue&
                   operator=           (DetectNoDataByValue&&)=default;

protected:

                   DetectNoDataByValue ()=delete;

private:

    Mask const&    _mask;

    value_type     _no_data_value;

};


template<
    typename Mask>
inline DetectNoDataByValue<Mask>::
        DetectNoDataByValue(
    Mask const& mask)

    : _mask(mask),
      _no_data_value(no_data_value(_mask))

{
}


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
    return is_equal(get(_mask, index), _no_data_value);
}


template<
    typename Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index1,
    size_t index2) const
{
    return get(_mask, index1, index2) == _no_data_value;
}


template<
    typename Mask>
inline bool DetectNoDataByValue<Mask>::is_no_data(
    size_t index1,
    size_t index2,
    size_t index3) const
{
    return get(_mask, index1, index2, index3) == _no_data_value;
}

} // namespace algorithm
} // namespace fern
