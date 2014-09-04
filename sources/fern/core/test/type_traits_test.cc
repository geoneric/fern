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
    BOOST_CHECK_EQUAL(fern::TypeTraits<fern::String>::value_types.count(), 1u);
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_types,
        fern::ValueTypes::UINT8);
    BOOST_CHECK_EQUAL(fern::TypeTraits<uint8_t>::value_types.to_string(),
        fern::String("Uint8"));
}

BOOST_AUTO_TEST_SUITE_END()
