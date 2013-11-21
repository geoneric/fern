#define BOOST_TEST_MODULE fern compiler
#include <boost/test/unit_test.hpp>
#include "fern/compiler/compiler.h"


BOOST_AUTO_TEST_SUITE(compiler)

BOOST_AUTO_TEST_CASE(constructor)
{
    fern::Compiler compiler("h", "cc");
}

BOOST_AUTO_TEST_SUITE_END()
