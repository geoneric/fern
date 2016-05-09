// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern wkt scope
#include <iostream>
#define BOOST_SPIRIT_X3_DEBUG
#include <boost/test/unit_test.hpp>
#include "fern/wkt/scope.h"


namespace x3 = boost::spirit::x3;


BOOST_AUTO_TEST_CASE(example_from_spec)
{
    std::string wkt =
        R"(SCOPE["Large scale topographic mapping and cadastre."])";
    std::cout << wkt << std::endl;

    auto first = wkt.begin();
    auto last = wkt.end();

    bool result = x3::phrase_parse(first, last, fern::wkt::scope, x3::space);
    BOOST_CHECK(first == last);
    BOOST_CHECK(result);
}
