#define BOOST_TEST_MODULE ranally interpreter
#include <boost/test/unit_test.hpp>
#include "ranally/core/io_error.h"
#include "ranally/core/parse_error.h"
#include "ranally/core/validate_error.h"
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
    std::vector<ranally::String> valid_files = {
        "valid-1.ran",
        "valid-2.ran"
    };

    for(ranally::String const& filename: valid_files) {
        vertex = interpreter.parse_file(filename);
        BOOST_CHECK(vertex);
    }

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


BOOST_AUTO_TEST_CASE(validate)
{
    ranally::Interpreter interpreter;
    ranally::ScriptVertexPtr vertex;

    vertex = interpreter.parse_file("valid-1.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(ranally::ValidateError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "valid-1.ran:3:4: Undefined identifier: b");
    }

    vertex = interpreter.parse_file("valid-2.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(ranally::ValidateError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            // TODO unknown function
            "valid-2.ran:x:y: z");
    }
}

BOOST_AUTO_TEST_SUITE_END()
