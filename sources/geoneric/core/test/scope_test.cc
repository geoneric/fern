#define BOOST_TEST_MODULE geoneric core
#include <boost/any.hpp>
#include <boost/test/unit_test.hpp>
#include "geoneric/core/scope.h"


BOOST_AUTO_TEST_SUITE(scope)


BOOST_AUTO_TEST_CASE(general)
{
    {
        geoneric::Scope<int> scope;
        BOOST_CHECK(!scope.has_value("a"));

        scope.set_value("a", 5);
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(scope.value("a"), 5);

        scope.set_value("a", 6);
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(scope.value("a"), 6);
    }

    {
        geoneric::Scope<boost::any> scope;
        BOOST_CHECK(!scope.has_value("a"));

        scope.set_value("a", boost::any(5));
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(boost::any_cast<int const&>(scope.value("a")), 5);

        scope.set_value("a", boost::any(6));
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(boost::any_cast<int const&>(scope.value("a")), 6);
    }
}


BOOST_AUTO_TEST_SUITE_END()
