#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
#include "geoneric/feature/core/array_value.h"
#include "geoneric/feature/core/box.h"
#include "geoneric/feature/core/point.h"
#include "geoneric/feature/core/spatial_attribute.h"
#include "geoneric/feature/core/spatial_domain.h"


BOOST_AUTO_TEST_SUITE(spatial_attribute)

BOOST_AUTO_TEST_CASE(int_per_box)
{
    // An integer value is stored per box.
    typedef geoneric::Point<double, 2> Point;
    typedef geoneric::Box<Point> Box;
    typedef geoneric::SpatialDomain<Box> BoxDomain;
    typedef int Value;
    typedef geoneric::SpatialAttribute<BoxDomain, Value> BoxesAttribute;

    Point south_west;
    geoneric::set<0>(south_west, 1.1);
    geoneric::set<1>(south_west, 2.2);
    Point north_east;
    geoneric::set<0>(north_east, 3.3);
    geoneric::set<1>(north_east, 4.4);
    Box box(south_west, north_east);

    Value value = 5;

    BoxesAttribute attribute;
    BOOST_CHECK(attribute.empty());

    BoxesAttribute::GID gid = attribute.add(box, value);
    BOOST_CHECK(!attribute.empty());
    BOOST_CHECK_EQUAL(attribute.domain().size(), 1u);
    BOOST_CHECK_EQUAL(attribute.values().value(gid), value);
}


BOOST_AUTO_TEST_CASE(array_per_box)
{
    // A pointer to a 2D array is stored as value per box.
    typedef geoneric::Point<double, 2> Point;
    typedef geoneric::Box<Point> Box;
    typedef geoneric::SpatialDomain<Box> BoxDomain;
    typedef geoneric::ArrayValue<int, 2> Value;
    typedef std::shared_ptr<Value> ValuePtr;
    typedef geoneric::SpatialAttribute<BoxDomain, ValuePtr> BoxesAttribute;

    Point south_west;
    geoneric::set<0>(south_west, 1.1);
    geoneric::set<1>(south_west, 2.2);
    Point north_east;
    geoneric::set<0>(north_east, 3.3);
    geoneric::set<1>(north_east, 4.4);
    Box box(south_west, north_east);

    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    ValuePtr grid(new Value(geoneric::extents[nr_rows][nr_cols]));
    (*grid)[0][0] = -2;
    (*grid)[0][1] = -1;
    (*grid)[1][0] = 0;
    (*grid)[1][1] = 999;
    (*grid)[2][0] = 1;
    (*grid)[2][1] = 2;

    BoxesAttribute attribute;
    BOOST_CHECK(attribute.empty());

    BoxesAttribute::GID gid = attribute.add(box, grid);
    BOOST_CHECK(!attribute.empty());
    BOOST_CHECK_EQUAL(attribute.domain().size(), 1u);
    BOOST_CHECK(*attribute.values().value(gid) == *grid);
}

BOOST_AUTO_TEST_SUITE_END()
