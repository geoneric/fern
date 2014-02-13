#define BOOST_TEST_MODULE fern operation_core
#include <boost/test/unit_test.hpp>
#include "fern/core/typelist.h"


BOOST_AUTO_TEST_SUITE(typelist)

BOOST_AUTO_TEST_CASE(typelist)
{
    namespace fc = fern::core;

    typedef fc::Typelist<> Types1;
    BOOST_CHECK_EQUAL(fc::size<Types1>::value, 0);

    typedef fc::Typelist<uint8_t, uint16_t, uint32_t> Types2;
    BOOST_CHECK_EQUAL(fc::size<Types2>::value, 3);

    typedef fc::push_front<uint64_t, Types2>::type Types3;
    BOOST_CHECK_EQUAL(fc::size<Types3>::value, 4);

    BOOST_CHECK((std::is_same<fc::at<0, Types3>::type, uint64_t>::value));
    BOOST_CHECK((std::is_same<fc::at<3, Types3>::type, uint32_t>::value));
    BOOST_CHECK_EQUAL((fc::find<uint64_t, Types3>::value), 0);
    BOOST_CHECK_EQUAL((fc::find<uint32_t, Types3>::value), 3);

    typedef fc::pop_front<Types3>::type Types4;
    BOOST_CHECK_EQUAL(fc::size<Types4>::value, 3);
    BOOST_CHECK((std::is_same<fc::at<0, Types4>::type, uint8_t>::value));
}

BOOST_AUTO_TEST_SUITE_END()
