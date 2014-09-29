#define BOOST_TEST_MODULE fern algorithm core
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/core/index_range.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(index_range)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fa::IndexRange range;

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 0);
        BOOST_CHECK_EQUAL(range.end(), 0);
    }

    {
        fa::IndexRange range(3, 6);

        BOOST_CHECK(!range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 3);
        BOOST_CHECK_EQUAL(range.end(), 6);
    }

    {
        fa::IndexRange range(5, 5);

        BOOST_CHECK(range.empty());
        BOOST_CHECK_EQUAL(range.begin(), 5);
        BOOST_CHECK_EQUAL(range.end(), 5);
    }
}


BOOST_AUTO_TEST_CASE(equality)
{
    BOOST_CHECK_EQUAL(fa::IndexRange(), fa::IndexRange());
    BOOST_CHECK_EQUAL(fa::IndexRange(5, 6), fa::IndexRange(5, 6));

    // Both are empty, but the same.
    BOOST_CHECK_EQUAL(fa::IndexRange(5, 5), fa::IndexRange(5, 5));

    BOOST_CHECK(fa::IndexRange(5, 6) != fa::IndexRange(5, 7));

    // Both are empty, but different.
    BOOST_CHECK(fa::IndexRange(5, 5) != fa::IndexRange(6, 6));
}

BOOST_AUTO_TEST_SUITE_END()
