#define BOOST_TEST_MODULE geoneric algorithm
#include <boost/test/unit_test.hpp>
#include <array>
#include "geoneric/algorithm/plus.h"
#include "geoneric/raster.h"


BOOST_AUTO_TEST_SUITE(plus)

BOOST_AUTO_TEST_CASE(argument_and_expression_types)
{
    // int + int -> int
    {
        int argument1 = 3;
        int argument2 = 4;
        int result = 0;
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result, 7);
    }

    // int + int[2] -> int[2]
    {
        int argument1 = 3;
        int argument2[2] = {4, 5};
        int result[2] = {0, 0};
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 8);
    }

    // int[2] + int -> int[2]
    {
        int argument1[2] = {3, 4};
        int argument2 = 4;
        int result[2] = {0, 0};
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 8);
    }

    // int[2] + int[2] -> int[2]
    {
        int argument1[2] = {3, 4};
        int argument2[2] = {4, 5};
        int result[2] = {0, 0};
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 9);
    }

    // int + std::array<int, 2> -> int[2]
    {
        int argument1 = 3;
        std::array<int, 2> argument2;
        argument2[0] = 4;
        argument2[1] = 5;
        int result[2] = {0, 0};
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 8);
    }

    // int[2] + std::array<int, 2> -> std::vector<int>(2, 0)
    {
        int argument1[2] = {3, 4};
        std::array<int, 2> argument2;
        argument2[0] = 4;
        argument2[1] = 5;
        std::vector<int> result(2, 0);
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 9);
    }

    // int[2] + int -> std::array<int, 2>
    {
        int argument1[2] = {3, 4};
        int argument2 = 4;
        int result[2] = {0, 0};
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result[0], 7);
        BOOST_CHECK_EQUAL(result[1], 8);
    }

    // Raster + Raster -> Raster
    {
        geoneric::Raster<int, 1, 2> argument1;
        argument1.set(0, 0, 3);
        argument1.set(0, 1, 4);
        geoneric::Raster<int, 1, 2> argument2;
        argument2.set(0, 0, 4);
        argument2.set(0, 1, 5);
        geoneric::Raster<int, 2, 3> result(0);
        geoneric::algorithm::plus(argument1, argument2, result);
        BOOST_CHECK_EQUAL(result.get(0, 0), 7);
        BOOST_CHECK_EQUAL(result.get(0, 1), 9);
    }

    // // int[1][2] + int -> int[2]
    // {
    //     int argument1[1][2];
    //     argument1[0][0] = 3;
    //     argument1[0][1] = 4;
    //     int argument2 = 4;
    //     int result[2] = {0, 0};
    //     geoneric::algorithm::plus(argument1, argument2, result);
    //     // BOOST_CHECK_EQUAL(result[0], 7);
    //     // BOOST_CHECK_EQUAL(result[1], 8);
    // }
}

BOOST_AUTO_TEST_SUITE_END()
