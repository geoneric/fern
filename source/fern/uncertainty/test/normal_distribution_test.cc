// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern uncertainty
#include <boost/test/unit_test.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "fern/uncertainty/normal_distribution.h"


BOOST_AUTO_TEST_SUITE(normal_distribution)

BOOST_AUTO_TEST_CASE(constructor)
{
    fern::NormalDistribution<double> distribution(5.0, 2.5);

    BOOST_CHECK_CLOSE(distribution.mean(), 5.0, 0.001);
    BOOST_CHECK_CLOSE(distribution.standard_deviation(), 2.5, 0.001);

    // boost::random::mt19937 random_number_generator;

    // for(size_t i = 0; i < 1000; ++i) {
    //     std::cout << i << " " << distribution(random_number_generator) << std::endl;
    // }

    // BOOST_CHECK(false);
}

BOOST_AUTO_TEST_SUITE_END()
