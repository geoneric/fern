#define BOOST_TEST_MODULE ranally interpreter
#include <boost/test/unit_test.hpp>
#include "ranally/core/io_error.h"
#include "ranally/core/parse_error.h"
#include "ranally/core/validate_error.h"
#include "ranally/operation/core/attribute_argument.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/interpreter/execute_visitor.h"
#include "ranally/interpreter/interpreter.h"


BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(parse_string)
{
    ranally::Interpreter interpreter;
    ranally::ModuleVertexPtr vertex;

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
            "Error parsing <string>:1:7:a = b c: invalid syntax");
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    ranally::Interpreter interpreter;
    ranally::ModuleVertexPtr vertex;

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
            "Error parsing invalid-1.ran:1:7:a = b c: invalid syntax");
    }

    // Unreadable file, io error.
    try {
        interpreter.parse_file("valid-1_unreadable.ran");
        BOOST_CHECK(false);
    }
    catch(ranally::IOError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error handling valid-1_unreadable.ran: Permission denied");
    }
}


BOOST_AUTO_TEST_CASE(validate)
{
    ranally::Interpreter interpreter;
    ranally::ModuleVertexPtr vertex;

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
            "valid-2.ran:4:4: Undefined operation: does_not_exist");
    }

    // String with unknown operation.
    vertex = interpreter.parse_string("a = blah(5)");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(ranally::ValidateError const& exception) {
        ranally::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "<string>:1:4: Undefined operation: blah");
    }
}


BOOST_AUTO_TEST_CASE(execute)
{
    ranally::Interpreter interpreter;
    ranally::ModuleVertexPtr tree;

    struct TestAbsResult {
        void operator()(ranally::Interpreter& interpreter) {
            std::stack<std::shared_ptr<ranally::Argument>> stack(
                interpreter.stack());
            BOOST_CHECK_EQUAL(stack.size(), 1u);

            std::shared_ptr<ranally::Argument> const& argument(stack.top());
            BOOST_REQUIRE_EQUAL(argument->argument_type(),
                ranally::ArgumentType::AT_ATTRIBUTE);

            std::shared_ptr<ranally::AttributeArgument> const&
                attribute_argument(
                    std::dynamic_pointer_cast<ranally::AttributeArgument>(
                        argument));
            BOOST_REQUIRE(attribute_argument);

            std::shared_ptr<ranally::Attribute> const& attribute(
                attribute_argument->attribute());

            BOOST_REQUIRE_EQUAL(attribute->data_type(), ranally::DT_SCALAR);
            BOOST_REQUIRE_EQUAL(attribute->value_type(), ranally::VT_INT64);

            std::shared_ptr<ranally::ScalarAttribute<int64_t>> scalar_attribute(
                std::dynamic_pointer_cast<ranally::ScalarAttribute<int64_t>>(
                    attribute));
            BOOST_REQUIRE(scalar_attribute);

            BOOST_CHECK_EQUAL((*scalar_attribute->value())(), 5);
        }
    };

    // Calculate abs(-5) and leave the result on the stack for testing.
    {
        tree = interpreter.parse_string("abs(-5)");
        BOOST_REQUIRE(tree);
        interpreter.execute(tree);
        TestAbsResult()(interpreter);
    }

    // Calculate abs(-5) and leave the result on the stack for testing.
    // Call a user defined function that does the calculation.
    {
        tree = interpreter.parse_string(
            "def do_abs(value):\n"
            "    return abs(value)\n"
            "do_abs(-5)");
        BOOST_REQUIRE(tree);
        interpreter.execute(tree);
        TestAbsResult()(interpreter);
    }




    // ranally::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // ranally::Stack const& stack(execute_visitor.stack());

    // BOOST_CHECK_EQUAL(symbol_table.size(), 1u);
    // BOOST_CHECK(symbol_table.has_value("a"));
    // BOOST_CHECK(symbol_table.value("a").type() == typeid(int32_t));

    // BOOST_CHECK_EQUAL(stack.size(), 0u);


    // tree = interpreter.parse_string("a = 5 + 6");
    // BOOST_REQUIRE(tree);
    // TODO Undefined identifier a, because operation doesn't calculate correct
    //      data_type/value_type yet. See annotate_visitor_test.
    // interpreter.validate(tree);
    // interpreter.execute(tree);

    // ranally::ExecuteVisitor execute_visitor;
    // tree->Accept(execute_visitor);

    // ranally::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // ranally::Stack const& stack(execute_visitor.stack());

    // BOOST_CHECK_EQUAL(symbol_table.size(), 1u);
    // BOOST_CHECK(symbol_table.has_value("a"));
    // BOOST_CHECK(symbol_table.value("a").type() == typeid(int32_t));

    // BOOST_CHECK_EQUAL(stack.size(), 0u);
}

BOOST_AUTO_TEST_SUITE_END()
