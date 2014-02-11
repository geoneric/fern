#define BOOST_TEST_MODULE fern expression_tree raster
#include <boost/multi_array.hpp>
#include <boost/test/unit_test.hpp>
#include "fern/expression_tree/raster.h"


BOOST_AUTO_TEST_SUITE(raster)

BOOST_AUTO_TEST_CASE(use_cases)
{
    // size_t const nr_rows = 3000;
    // size_t const nr_cols = 4000;

    // // A raster modelled by a 2D Boost multi array.
    // {
    //     typedef boost::multi_array<int32_t, 2> Array;
    //     auto extents(boost::extents[nr_rows][nr_cols]);
    //     Array array(extents);

    //     fern::expression_tree::Raster<Array> raster(array);

    //     BOOST_CHECK(static_cast<Array const&>(raster) == array);
    //     BOOST_CHECK(Array(static_cast<Array const&>(raster)) == array);
    // }

    // // A raster modelled by a 2D C array.
    // {
    //     typedef int32_t* Array;
    //     std::unique_ptr<int32_t> array(new int32_t[nr_rows * nr_cols]);

    //     fern::Raster<Array> raster(array);
    // }
}

BOOST_AUTO_TEST_SUITE_END()
