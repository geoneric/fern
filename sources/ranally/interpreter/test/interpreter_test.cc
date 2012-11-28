#define BOOST_TEST_MODULE ranally interpreter
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/exception.h"
#include "ranally/interpreter/interpreter.h"


BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(parse_string)
{
    ranally::Interpreter interpreter;

    ranally::ScriptVertexPtr vertex = interpreter.parse_string("a = b + c");

    try {
        interpreter.parse_string("a = b c");
    }
    catch(ranally::ParseError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error while parsing: invalid syntax\n"
            "1:7: a = b c");
    }
}

BOOST_AUTO_TEST_SUITE_END()
