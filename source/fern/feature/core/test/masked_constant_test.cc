#define BOOST_TEST_MODULE fern feature core
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/masked_constant.h"


BOOST_AUTO_TEST_SUITE(masked_constant)

BOOST_AUTO_TEST_CASE(masked_constant)
{
    {
        fern::MaskedConstant<int32_t> c_int32_t;
        BOOST_CHECK(!c_int32_t.mask());
        BOOST_CHECK_EQUAL(c_int32_t.value(), 0);

        c_int32_t = fern::MaskedConstant<int32_t>(5);
        BOOST_CHECK(!c_int32_t.mask());
        BOOST_CHECK_EQUAL(c_int32_t.value(), 5);
    }

    {
        fern::MaskedConstant<int32_t> c_int32_t(5);

        BOOST_CHECK(!c_int32_t.mask());
        BOOST_CHECK_EQUAL(c_int32_t.value(), 5);

        c_int32_t.value() = 6;
        BOOST_CHECK(!c_int32_t.mask());
        BOOST_CHECK_EQUAL(c_int32_t.value(), 6);

        c_int32_t.mask() = true;
        BOOST_CHECK(c_int32_t.mask());
        BOOST_CHECK_EQUAL(c_int32_t.value(), 6);
    }
}

BOOST_AUTO_TEST_SUITE_END()
