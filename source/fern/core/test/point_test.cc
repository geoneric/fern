#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/point_traits.h"


BOOST_AUTO_TEST_SUITE(point)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fern::Point<int, 1> point;
        BOOST_CHECK_EQUAL(fern::get<0>(point), 0);
    }

    {
        fern::Point<int, 1> point(5);
        BOOST_CHECK_EQUAL(fern::get<0>(point), 5);
    }

    {
        fern::Point<int, 2> point;
        BOOST_CHECK_EQUAL(fern::get<0>(point), 0);
        BOOST_CHECK_EQUAL(fern::get<1>(point), 0);
    }

    {
        fern::Point<int, 2> point(5, 6);
        BOOST_CHECK_EQUAL(fern::get<0>(point), 5);
        BOOST_CHECK_EQUAL(fern::get<1>(point), 6);
    }

    {
        fern::Point<int, 3> point;
        BOOST_CHECK_EQUAL(fern::get<0>(point), 0);
        BOOST_CHECK_EQUAL(fern::get<1>(point), 0);
        BOOST_CHECK_EQUAL(fern::get<2>(point), 0);
    }

    {
        fern::Point<int, 3> point(5, 6, 7);
        BOOST_CHECK_EQUAL(fern::get<0>(point), 5);
        BOOST_CHECK_EQUAL(fern::get<1>(point), 6);
        BOOST_CHECK_EQUAL(fern::get<2>(point), 7);
    }
}


BOOST_AUTO_TEST_CASE(copy)
{
    {
        fern::Point<int, 2> point1(5, 6);
        auto point2 = point1;
        BOOST_CHECK_EQUAL(fern::get<0>(point2), 5);
        BOOST_CHECK_EQUAL(fern::get<1>(point2), 6);
    }

    {
        fern::Point<int, 2> point1(5, 6);
        auto point2(point1);
        BOOST_CHECK_EQUAL(fern::get<0>(point2), 5);
        BOOST_CHECK_EQUAL(fern::get<1>(point2), 6);
    }
}

BOOST_AUTO_TEST_SUITE_END()
