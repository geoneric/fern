#define BOOST_TEST_MODULE geoneric interpreter
#include <boost/test/unit_test.hpp>
#include "geoneric/core/io_error.h"
#include "geoneric/core/parse_error.h"
#include "geoneric/core/validate_error.h"
#include "geoneric/feature/core/constant_attribute.h"
#include "geoneric/operation/core/attribute_argument.h"
#include "geoneric/interpreter/execute_visitor.h"
#include "geoneric/interpreter/interpreter.h"


BOOST_AUTO_TEST_SUITE(interpreter)

BOOST_AUTO_TEST_CASE(parse_string)
{
    geoneric::Interpreter interpreter;
    geoneric::ModuleVertexPtr vertex;

    // String with valid statements, should succeed.
    std::vector<geoneric::String> valid_statements = {
        "a = b + c",
        "a",
        "b + c",
        "",
        "# comment"
    };

    for(geoneric::String const& statement: valid_statements) {
        vertex = interpreter.parse_string(statement);
        BOOST_CHECK(vertex);
    }

    // String with invalid statement, syntax error with location information.
    try {
        interpreter.parse_string("a = b c");
        BOOST_CHECK(false);
    }
    catch(geoneric::ParseError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error parsing <string>:1:7:a = b c: invalid syntax");
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    geoneric::Interpreter interpreter;
    geoneric::ModuleVertexPtr vertex;

    // File with valid statement, should succeed.
    std::vector<geoneric::String> valid_files = {
        "valid-1.ran",
        "valid-2.ran"
    };

    for(geoneric::String const& filename: valid_files) {
        vertex = interpreter.parse_file(filename);
        BOOST_CHECK(vertex);
    }

    // File with invalid statement, syntax error with location information.
    try {
        interpreter.parse_file("invalid-1.ran");
        BOOST_CHECK(false);
    }
    catch(geoneric::ParseError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error parsing invalid-1.ran:1:7:a = b c: invalid syntax");
    }

    // Unreadable file, io error.
    try {
        interpreter.parse_file("valid-1_unreadable.ran");
        BOOST_CHECK(false);
    }
    catch(geoneric::IOError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error handling valid-1_unreadable.ran: Permission denied");
    }
}


BOOST_AUTO_TEST_CASE(validate)
{
    geoneric::Interpreter interpreter;
    geoneric::ModuleVertexPtr vertex;

    vertex = interpreter.parse_file("valid-1.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(geoneric::ValidateError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "valid-1.ran:3:4: Undefined identifier: b");
    }

    vertex = interpreter.parse_file("valid-2.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(geoneric::ValidateError const& exception) {
        geoneric::String message = exception.message();
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
    catch(geoneric::ValidateError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "<string>:1:4: Undefined operation: blah");
    }

    // Verify that calling user-defined operation doesn't throw.
    {
        vertex = interpreter.parse_string(u8R"(
def foo():
    return

foo()
)");
        BOOST_REQUIRE(vertex);
        // TODO Make sure user-defined operations are detected and handled
        //      like built-in ones.
        BOOST_CHECK_NO_THROW(interpreter.validate(vertex));
    }

    // Call user-defined operation with wrong number of arguments.
    {
        vertex = interpreter.parse_string(u8R"(
def foo():
    return

foo(5)
)");
        try {
            interpreter.validate(vertex);
            BOOST_CHECK(false);
        }
        catch(geoneric::ValidateError const& exception) {
            geoneric::String message = exception.message();
            // TODO Update message in test.
            BOOST_CHECK_EQUAL(message,
                "<string>:1:4: Undefined operation: blah");
        }
    }
}


BOOST_AUTO_TEST_CASE(execute)
{
    geoneric::Interpreter interpreter;
    geoneric::ModuleVertexPtr tree;

    struct TestAbsResult {
        void operator()(geoneric::Interpreter& interpreter) {
            std::stack<std::shared_ptr<geoneric::Argument>> stack(
                interpreter.stack());
            BOOST_CHECK_EQUAL(stack.size(), 1u);

            std::shared_ptr<geoneric::Argument> const& argument(stack.top());
            BOOST_REQUIRE_EQUAL(argument->argument_type(),
                geoneric::ArgumentType::AT_ATTRIBUTE);

            std::shared_ptr<geoneric::AttributeArgument> const&
                attribute_argument(
                    std::dynamic_pointer_cast<geoneric::AttributeArgument>(
                        argument));
            BOOST_REQUIRE(attribute_argument);
            BOOST_REQUIRE_EQUAL(attribute_argument->data_type(),
                geoneric::DT_SCALAR);
            BOOST_REQUIRE_EQUAL(attribute_argument->value_type(),
                geoneric::VT_INT64);

            std::shared_ptr<geoneric::Attribute> const& attribute(
                attribute_argument->attribute());

            std::shared_ptr<geoneric::ConstantAttribute<int64_t>>
                constant_attribute(std::dynamic_pointer_cast<
                    geoneric::ConstantAttribute<int64_t>>(attribute));
            BOOST_REQUIRE(constant_attribute);
            BOOST_CHECK_EQUAL(constant_attribute->values().value(), 5);
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




    // geoneric::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // geoneric::Stack const& stack(execute_visitor.stack());

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

    // geoneric::ExecuteVisitor execute_visitor;
    // tree->Accept(execute_visitor);

    // geoneric::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // geoneric::Stack const& stack(execute_visitor.stack());

    // BOOST_CHECK_EQUAL(symbol_table.size(), 1u);
    // BOOST_CHECK(symbol_table.has_value("a"));
    // BOOST_CHECK(symbol_table.value("a").type() == typeid(int32_t));

    // BOOST_CHECK_EQUAL(stack.size(), 0u);
}

BOOST_AUTO_TEST_SUITE_END()
