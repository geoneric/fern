// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/language/feature/core/array_value.h"
#include "fern/language/feature/core/constant_attribute.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_SUITE(constant_attribute)

BOOST_AUTO_TEST_CASE(int_)
{
    // An integer value is stored.
    using Value = int;
    using Attribute = fl::ConstantAttribute<Value>;

    Value value = 5;

    Attribute attribute;
    attribute.set(value);
    BOOST_CHECK_EQUAL(attribute.values().value(), value);
}


BOOST_AUTO_TEST_CASE(array_per_box)
{
    // A pointer to a 2D array is stored.
    using Value = fl::ArrayValue<int, 2>;
    using ValuePtr = std::shared_ptr<fl::ArrayValue<int, 2>>;
    using Attribute = fl::ConstantAttribute<ValuePtr>;

    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    ValuePtr grid = std::make_shared<Value>(fern::extents[nr_rows][nr_cols]);
    (*grid)[0][0] = -2;
    (*grid)[0][1] = -1;
    (*grid)[1][0] = 0;
    (*grid)[1][1] = 999;
    (*grid)[2][0] = 1;
    (*grid)[2][1] = 2;

    Attribute attribute;
    attribute.set(grid);
    // TODO BOOST_CHECK(*attribute.values().value() == *grid);
}

BOOST_AUTO_TEST_SUITE_END()
