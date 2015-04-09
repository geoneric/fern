// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
    //     using Array = boost::multi_array<int32_t, 2>;
    //     auto extents(boost::extents[nr_rows][nr_cols]);
    //     Array array(extents);

    //     fern::expression_tree::Raster<Array> raster(array);

    //     BOOST_CHECK(static_cast<Array const&>(raster) == array);
    //     BOOST_CHECK(Array(static_cast<Array const&>(raster)) == array);
    // }

    // // A raster modelled by a 2D C array.
    // {
    //     using Array = int32_t*;
    //     std::unique_ptr<int32_t> array(new int32_t[nr_rows * nr_cols]);

    //     fern::Raster<Array> raster(array);
    // }
}

BOOST_AUTO_TEST_SUITE_END()
