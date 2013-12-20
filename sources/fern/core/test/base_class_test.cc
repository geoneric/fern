#define BOOST_TEST_MODULE fern operation_core
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
    BOOST_CHECK((std::is_same<fern::base_class<pear_tag, pear_tag>::type,
        pear_tag>::value));
    BOOST_CHECK((std::is_same<fern::base_class<pear_tag, pome_tag>::type,
        pome_tag>::value));
    BOOST_CHECK((std::is_same<fern::base_class<pear_tag, citrus_tag>::type,
        pear_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, pome_tag, citrus_tag>::type,
        pome_tag>::value));
    BOOST_CHECK((std::is_same<
        fern::base_class<pear_tag, citrus_tag, pome_tag>::type,
        pome_tag>::value));
}

BOOST_AUTO_TEST_SUITE_END()
