#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/gdal/spatial_attribute.h"


BOOST_AUTO_TEST_SUITE(spatial_attribute)

BOOST_AUTO_TEST_CASE(spatial_attribute)
{
    // typedef geoneric::Point<double, 2> Point;
    // typedef geoneric::Box<Point> Box;
    // typedef geoneric::SpatialDomain<Box> SpatialDomain;

    // size_t const nr_rows = 2;
    // size_t const nr_cols = 3;
    // geoneric::ArrayValue<int, nr_rows, nr_cols> grid;


    // size_t const nr_cells = 3;
    // geoneric::Value<int, nr_cells> array;

    // array[0] = 3;
    // array[1] = 2;
    // array[2] = 1;

    // BOOST_CHECK_EQUAL(array.size(), nr_cells);
    // BOOST_CHECK_EQUAL(array[0], 3);
    // BOOST_CHECK_EQUAL(array[1], 2);
    // BOOST_CHECK_EQUAL(array[2], 1);
}

BOOST_AUTO_TEST_SUITE_END()
