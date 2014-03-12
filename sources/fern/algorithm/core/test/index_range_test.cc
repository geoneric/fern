#define BOOST_TEST_MODULE fern algorithm core
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/index_range.h"


BOOST_AUTO_TEST_SUITE(index_range)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fern::IndexRange range;

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 0);
        BOOST_CHECK_EQUAL(range.end(), 0);
    }

    {
        fern::IndexRange range(3, 6);

        BOOST_CHECK(!range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 3);
        BOOST_CHECK_EQUAL(range.end(), 6);
    }

    {
        fern::IndexRange range(5, 5);

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 5);
        BOOST_CHECK_EQUAL(range.end(), 5);
    }
}


BOOST_AUTO_TEST_CASE(equality)
{
    BOOST_CHECK_EQUAL(fern::IndexRange(), fern::IndexRange());
    BOOST_CHECK_EQUAL(fern::IndexRange(5, 6), fern::IndexRange(5, 6));

    // Both are empty, but the same.
    BOOST_CHECK_EQUAL(fern::IndexRange(5, 5), fern::IndexRange(5, 5));

    BOOST_CHECK(fern::IndexRange(5, 6) != fern::IndexRange(5, 7));

    // Both are empty, but different.
    BOOST_CHECK(fern::IndexRange(5, 5) != fern::IndexRange(6, 6));
}

BOOST_AUTO_TEST_SUITE_END()
