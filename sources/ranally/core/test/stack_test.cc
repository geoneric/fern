#define BOOST_TEST_MODULE ranally core
#include <boost/test/unit_test.hpp>
#include "ranally/core/stack.h"


BOOST_AUTO_TEST_SUITE(stack)

BOOST_AUTO_TEST_CASE(stack)
{
    ranally::Stack stack;

    BOOST_CHECK_EQUAL(stack.size(), 0u);
    BOOST_CHECK(stack.empty());

    stack.push(std::string("5"));
    stack.push(int(5));

    BOOST_CHECK_EQUAL(stack.size(), 2u);
    BOOST_CHECK(!stack.empty());

    BOOST_CHECK_EQUAL(stack.top<int>(), int(5));
    stack.pop();

    BOOST_CHECK_EQUAL(stack.size(), 1u);
    BOOST_CHECK(!stack.empty());

    BOOST_CHECK_EQUAL(stack.top<std::string>(), std::string("5"));
    stack.pop();

    BOOST_CHECK_EQUAL(stack.size(), 0u);
    BOOST_CHECK(stack.empty());
}

BOOST_AUTO_TEST_SUITE_END()
