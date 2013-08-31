#define BOOST_TEST_MODULE ranally uncertainty
#include <boost/test/unit_test.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "ranally/uncertainty/normal_distribution.h"


BOOST_AUTO_TEST_SUITE(normal_distribution)

BOOST_AUTO_TEST_CASE(constructor)
{
    ranally::NormalDistribution<double> distribution(5.0, 2.5);

    BOOST_CHECK_CLOSE(distribution.mean(), 5.0, 0.001);
    BOOST_CHECK_CLOSE(distribution.standard_deviation(), 2.5, 0.001);

    // boost::random::mt19937 random_number_generator;

    // for(size_t i = 0; i < 1000; ++i) {
    //     std::cout << i << " " << distribution(random_number_generator) << std::endl;
    // }

    // BOOST_CHECK(false);
}

BOOST_AUTO_TEST_SUITE_END()
