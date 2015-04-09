// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/any.hpp>
#include <boost/test/unit_test.hpp>
#include "fern/core/scope.h"


BOOST_AUTO_TEST_SUITE(scope)


BOOST_AUTO_TEST_CASE(general)
{
    {
        fern::Scope<int> scope;
        BOOST_CHECK(!scope.has_value("a"));

        scope.set_value("a", 5);
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(scope.value("a"), 5);

        scope.set_value("a", 6);
        BOOST_CHECK(scope.has_value("a"));
        BOOST_CHECK_EQUAL(scope.value("a"), 6);
    }

    {
        fern::Scope<boost::any> scope;
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
