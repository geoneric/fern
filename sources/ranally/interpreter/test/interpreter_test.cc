#define BOOST_TEST_MODULE ranally interpreter
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/io_error.h"
#include "ranally/core/parse_error.h"
#include "ranally/interpreter/interpreter.h"


BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(parse_string)
{
    ranally::Interpreter interpreter;
    ranally::ScriptVertexPtr vertex;

    // String with valid statements, should succeed.
    std::vector<ranally::String> valid_statements = {
        "a = b + c",
        "a",
        "b + c",
        "",
        "# comment"
    };

    for(ranally::String const& statement: valid_statements) {
        vertex = interpreter.parse_string(statement);
        BOOST_CHECK(vertex);
    }

    // String with invalid statement, syntax error with location information.
    try {
        interpreter.parse_string("a = b c");
        BOOST_CHECK(false);
    }
    catch(ranally::ParseError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error while parsing: invalid syntax\n"
            "1:7: a = b c");
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    ranally::Interpreter interpreter;
    ranally::ScriptVertexPtr vertex;

    // File with valid statement, should succeed.
    vertex = interpreter.parse_file("valid-1.ran");
    BOOST_CHECK(vertex);

    // File with invalid statement, syntax error with location information.
    try {
        interpreter.parse_file("invalid-1.ran");
        BOOST_CHECK(false);
    }
    catch(ranally::ParseError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error while parsing file invalid-1.ran: invalid syntax\n"
            "1:7: a = b c");
    }

    // Unreadable file, io error.
    try {
        interpreter.parse_file("valid-1_unreadable.ran");
        BOOST_CHECK(false);
    }
    catch(ranally::IOError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error while handling file valid-1_unreadable.ran: "
            "Permission denied");
    }
}

BOOST_AUTO_TEST_SUITE_END()
