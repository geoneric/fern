#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/array_value.h"
#include "fern/feature/core/constant_attribute.h"


BOOST_AUTO_TEST_SUITE(constant_attribute)

BOOST_AUTO_TEST_CASE(int_)
{
    // An integer value is stored.
    typedef int Value;
    typedef fern::ConstantAttribute<Value> Attribute;

    Value value = 5;

    Attribute attribute;
    attribute.set(value);
    BOOST_CHECK_EQUAL(attribute.values().value(), value);
}


BOOST_AUTO_TEST_CASE(array_per_box)
{
    // A pointer to a 2D array is stored.
    typedef fern::ArrayValue<int, 2> Value;
    typedef std::shared_ptr<fern::ArrayValue<int, 2>> ValuePtr;
    typedef fern::ConstantAttribute<ValuePtr> Attribute;

    size_t const nr_rows = 3;
    size_t const nr_cols = 2;
    ValuePtr grid(new Value(fern::extents[nr_rows][nr_cols]));
    (*grid)[0][0] = -2;
    (*grid)[0][1] = -1;
    (*grid)[1][0] = 0;
    (*grid)[1][1] = 999;
    (*grid)[2][0] = 1;
    (*grid)[2][1] = 2;

    Attribute attribute;
    attribute.set(grid);
    BOOST_CHECK(*attribute.values().value() == *grid);
}

BOOST_AUTO_TEST_SUITE_END()
