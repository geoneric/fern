// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/base_class.h"


struct pome_tag {};
struct apple_tag: pome_tag {};
struct pear_tag: pome_tag {};

struct banana_tag {};

struct citrus_tag {};
struct orange_tag: citrus_tag {};
struct lemon_tag: citrus_tag {};
struct lime_tag: citrus_tag {};


BOOST_AUTO_TEST_SUITE(base_class)

BOOST_AUTO_TEST_CASE(base_class)
{
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, pear_tag>, pear_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, pome_tag>, pome_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, citrus_tag>, pear_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, pome_tag, citrus_tag>, pome_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, citrus_tag, pome_tag>, pome_tag>::value));
}

BOOST_AUTO_TEST_SUITE_END()
