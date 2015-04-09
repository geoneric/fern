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
#include "fern/core/stack.h"


BOOST_AUTO_TEST_SUITE(stack)

BOOST_AUTO_TEST_CASE(stack_1)
{
    fern::Stack stack;

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


BOOST_AUTO_TEST_CASE(stack_2)
{
    {
        fern::Stack stack;
        stack.push(std::string("5"));
        boost::any value = stack.top();
        BOOST_CHECK_EQUAL(boost::any_cast<std::string>(value),
            std::string("5"));
    }

    {
        // Push a shared pointer on the stack. The use count should be
        // increased by one. When copying the value from the stack, the
        // use count should again be increased by one.
        fern::Stack stack;

        // One shared pointer.
        std::shared_ptr<std::string> value1(std::make_shared<std::string>("5"));
        BOOST_CHECK_EQUAL(value1.use_count(), 1u);

        // Copy shared pointer to the stack.
        stack.push(value1);
        BOOST_CHECK_EQUAL(value1.use_count(), 2u);

        // Copy shared pointer from the stack.
        boost::any value2(stack.top());
        BOOST_CHECK_EQUAL(value1.use_count(), 3u);
    }
}

BOOST_AUTO_TEST_SUITE_END()
