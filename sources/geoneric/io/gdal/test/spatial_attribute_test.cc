#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/gdal/array_value.h"
#include "geoneric/io/gdal/box.h"
#include "geoneric/io/gdal/point.h"
#include "geoneric/io/gdal/spatial_attribute.h"
#include "geoneric/io/gdal/spatial_domain.h"


BOOST_AUTO_TEST_SUITE(spatial_attribute)

BOOST_AUTO_TEST_CASE(spatial_attribute)
{
    typedef geoneric::Point<double, 2> Point;
    typedef geoneric::Box<Point> Box;
    typedef geoneric::SpatialDomain<Box> BoxDomain;
    typedef geoneric::ArrayValue<int, 2> GridValue;
    typedef geoneric::SpatialAttribute<BoxDomain, GridValue> Raster;

    Point south_west;
    geoneric::set<0>(south_west, 1.1);
    geoneric::set<1>(south_west, 2.2);
    Point north_east;
    geoneric::set<0>(north_east, 3.3);
    geoneric::set<1>(north_east, 4.4);
    Box box(south_west, north_east);

    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    GridValue grid(geoneric::extents[nr_rows][nr_cols]);
    grid[0][0] = -2;
    grid[0][1] = -1;
    grid[1][0] = 0;
    grid[1][1] = 999;
    grid[2][0] = 1;
    grid[2][1] = 2;

    Raster raster;
    // Hier verder. Implement copy or go with pointers to grids. Maybe the
    // implementation doesn't need to change. We can just pass a pointer
    // type as template parameter of SpatialAttribute instead. In that case
    // a pointer will be copied instead.
    // raster.add(box, grid);
}

BOOST_AUTO_TEST_SUITE_END()
