// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core typelist
#include <boost/test/unit_test.hpp>
#include "fern/core/typelist.h"


BOOST_AUTO_TEST_CASE(typelist)
{
    namespace fc = fern::core;

    using Types1 = fc::Typelist<>;
    BOOST_CHECK_EQUAL(fc::size<Types1>::value, 0);

    using Types2 = fc::Typelist<uint8_t, uint16_t, uint32_t>;
    BOOST_CHECK_EQUAL(fc::size<Types2>::value, 3);

    using Types3 = fc::push_front<uint64_t, Types2>::type;
    BOOST_CHECK_EQUAL(fc::size<Types3>::value, 4);

    BOOST_CHECK((std::is_same<fc::at<0, Types3>::type, uint64_t>::value));
    BOOST_CHECK((std::is_same<fc::at<3, Types3>::type, uint32_t>::value));
    BOOST_CHECK_EQUAL((fc::find<uint64_t, Types3>::value), 0);
    BOOST_CHECK_EQUAL((fc::find<uint32_t, Types3>::value), 3);

    using Types4 = fc::pop_front<Types3>::type;
    BOOST_CHECK_EQUAL(fc::size<Types4>::value, 3);
    BOOST_CHECK((std::is_same<fc::at<0, Types4>::type, uint8_t>::value));
}
