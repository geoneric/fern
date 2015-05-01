// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"


BOOST_AUTO_TEST_SUITE(type_traits)

BOOST_AUTO_TEST_CASE(type_traits)
{
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_type,
        fern::VT_UINT8);
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_types.count(), 1u);
    BOOST_CHECK_EQUAL(fern::TypeTraits<double>::value_types.count(), 1u);
    BOOST_CHECK_EQUAL(fern::TypeTraits<std::string>::value_types.count(), 1u);
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_types,
        fern::ValueTypes::UINT8);
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_types.to_string(),
        "Uint8");
}


BOOST_AUTO_TEST_CASE(are_same)
{
    BOOST_CHECK( (fern::are_same<int>::value));
    BOOST_CHECK(!(fern::are_same<int, float>::value));
    BOOST_CHECK( (fern::are_same<int, int>::value));
    BOOST_CHECK( (fern::are_same<int, int, int>::value));
    BOOST_CHECK(!(fern::are_same<int, int, int, double>::value));
}

BOOST_AUTO_TEST_SUITE_END()
