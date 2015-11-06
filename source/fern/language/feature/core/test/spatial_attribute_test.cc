// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature spatial_attribute
#include <boost/test/unit_test.hpp>
#include "fern/language/feature/core/attributes.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(int_per_box)
{
    using FieldAttribute = fl::SpatialAttribute<fl::FieldDomain, int>;

    // // An integer value is stored per box.
    fl::d2::Point south_west;
    fern::set<0>(south_west, 1.1);
    fern::set<1>(south_west, 2.2);
    fl::d2::Point north_east;
    fern::set<0>(north_east, 3.3);
    fern::set<1>(north_east, 4.4);
    fl::d2::Box box(south_west, north_east);

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
    using Value = fl::d2::ArrayValue<int>;
    using ValuePtr = fl::d2::ArrayValuePtr<int>;
    using FieldAttribute = fl::SpatialAttribute<fl::FieldDomain, ValuePtr>;

    fl::d2::Point south_west;
    fern::set<0>(south_west, 1.1);
    fern::set<1>(south_west, 2.2);
    fl::d2::Point north_east;
    fern::set<0>(north_east, 3.3);
    fern::set<1>(north_east, 4.4);
    fl::d2::Box box(south_west, north_east);

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
    // TODO BOOST_CHECK(*attribute.values().value(gid) == *grid);
}
