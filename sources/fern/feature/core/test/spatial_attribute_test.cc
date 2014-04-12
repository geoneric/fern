#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/attributes.h"


BOOST_AUTO_TEST_SUITE(spatial_attribute)

BOOST_AUTO_TEST_CASE(int_per_box)
{
    using FieldAttribute = fern::SpatialAttribute<fern::FieldDomain, int>;

    // // An integer value is stored per box.
    fern::d2::Point south_west;
    fern::set<0>(south_west, 1.1);
    fern::set<1>(south_west, 2.2);
    fern::d2::Point north_east;
    fern::set<0>(north_east, 3.3);
    fern::set<1>(north_east, 4.4);
    fern::d2::Box box(south_west, north_east);

    int value = 5;

    FieldAttribute attribute;
    BOOST_CHECK(attribute.empty());

    FieldAttribute::GID gid = attribute.add(box, value);
    BOOST_CHECK(!attribute.empty());
    BOOST_CHECK_EQUAL(attribute.domain().size(), 1u);
    BOOST_CHECK_EQUAL(attribute.values().value(gid), value);
}


BOOST_AUTO_TEST_CASE(array_per_box)
{
    // A pointer to a 2D array is stored as value per box.
    using Value = fern::d2::ArrayValue<int>;
    using ValuePtr = fern::d2::ArrayValuePtr<int>;
    using FieldAttribute = fern::SpatialAttribute<fern::FieldDomain, ValuePtr>;

    fern::d2::Point south_west;
    fern::set<0>(south_west, 1.1);
    fern::set<1>(south_west, 2.2);
    fern::d2::Point north_east;
    fern::set<0>(north_east, 3.3);
    fern::set<1>(north_east, 4.4);
    fern::d2::Box box(south_west, north_east);

    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    ValuePtr grid(std::make_shared<Value>(fern::extents[nr_rows][nr_cols]));
    (*grid)[0][0] = -2;
    (*grid)[0][1] = -1;
    (*grid)[1][0] = 0;
    (*grid)[1][1] = 999;
    (*grid)[2][0] = 1;
    (*grid)[2][1] = 2;

    FieldAttribute attribute;
    BOOST_CHECK(attribute.empty());

    FieldAttribute::GID gid = attribute.add(box, grid);
    BOOST_CHECK(!attribute.empty());
    BOOST_CHECK_EQUAL(attribute.domain().size(), 1u);
    BOOST_CHECK(*attribute.values().value(gid) == *grid);
}

BOOST_AUTO_TEST_SUITE_END()
