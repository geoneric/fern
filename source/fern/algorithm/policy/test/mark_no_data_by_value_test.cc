// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/core/data_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/mark_no_data_by_value.h"


template<
    class T>
class Mask
{

public:

    using value_type = T;
    using reference = T&;
    using const_reference = T const&;

    Mask()
        : _values(nullptr)
    {
    }

    Mask(T* values)
        : _values(values)
    {
    }

    inline T const& get(
        size_t index) const
    {
        return _values[index];
    }

    inline T& get(
        size_t index)
    {
        return _values[index];
    }

private:

    T* _values;

};


template<
    class T>
T const& get(
    Mask<T> const& mask,
    size_t index)
{
    return mask.get(index);
}


template<
    class T>
T& get(
    Mask<T>& mask,
    size_t index)
{
    return mask.get(index);
}


BOOST_AUTO_TEST_SUITE(mark_no_data)

BOOST_AUTO_TEST_CASE(mark_no_data)
{
    int32_t values[] = { 5, 4, 3, 2, 1 };
    Mask<int32_t> mask(values);

    fern::algorithm::MarkNoDataByValue<Mask<int32_t>> policy(mask, 6);

    policy.mark_as_no_data(0);
    policy.mark_as_no_data(4);
    BOOST_CHECK_EQUAL(values[0], 6);
    BOOST_CHECK_EQUAL(values[1], 4);
    BOOST_CHECK_EQUAL(values[2], 3);
    BOOST_CHECK_EQUAL(values[3], 2);
    BOOST_CHECK_EQUAL(values[4], 6);
}

BOOST_AUTO_TEST_SUITE_END()
