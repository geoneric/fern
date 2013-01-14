#define BOOST_TEST_MODULE ranally feature
#include <boost/test/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/feature/attribute.h"
#include "ranally/feature/feature.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/feature/scalar_domain.h"
#include "ranally/feature/scalar_value.h"


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(feature)
{
    // Let's play with feature earth.
    ranally::Feature earth;

    // Add some scalar properties.
    std::shared_ptr<ranally::ScalarDomain> scalar_domain(
        new ranally::ScalarDomain());

    // standard gravity.
    std::shared_ptr<ranally::ScalarValue<double>> gravity_value(
        new ranally::ScalarValue<double>(9.8));
    std::shared_ptr<ranally::ScalarAttribute<double>> gravity_attribute(
        new ranally::ScalarAttribute<double>("gravity", scalar_domain,
        gravity_value));
    earth.add_attribute(gravity_attribute);

    // latin name
    std::shared_ptr<ranally::ScalarValue<ranally::String>> latin_name_value(
        new ranally::ScalarValue<ranally::String>("terra"));
    std::shared_ptr<ranally::ScalarAttribute<ranally::String>>
        latin_name_attribute(new ranally::ScalarAttribute<ranally::String>(
            "latin_name", scalar_domain, latin_name_value));
    earth.add_attribute(latin_name_attribute);

    // Test the earth feature's attributes.
    BOOST_CHECK_EQUAL(earth.nr_attributes(), 2u);

    // TODO

}

BOOST_AUTO_TEST_SUITE_END()
