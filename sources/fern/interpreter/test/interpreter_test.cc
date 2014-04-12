#define BOOST_TEST_MODULE fern interpreter
#include <boost/test/unit_test.hpp>
#include "fern/core/io_error.h"
#include "fern/core/parse_error.h"
#include "fern/core/type_traits.h"
#include "fern/core/validate_error.h"
#include "fern/feature/core/attributes.h"
#include "fern/io/drivers.h"
#include "fern/io/io_client.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/operation/core/feature_argument.h"
#include "fern/interpreter/data_sources.h"
#include "fern/interpreter/data_syncs.h"
#include "fern/interpreter/execute_visitor.h"
#include "fern/interpreter/interpreter.h"


class Support:
    public fern::IOClient
{

public:

    Support()
        : fern::IOClient()
    {
    }

};


template<
    class T>
struct TestConstant {
    void operator()(
            fern::Interpreter& interpreter,
            T result)
    {
        std::stack<std::shared_ptr<fern::Argument>> stack(
            interpreter.stack());
        BOOST_CHECK_EQUAL(stack.size(), 1u);

        std::shared_ptr<fern::Argument> const& argument(stack.top());
        BOOST_REQUIRE_EQUAL(argument->argument_type(),
            fern::ArgumentType::AT_ATTRIBUTE);

        std::shared_ptr<fern::AttributeArgument> const&
            attribute_argument(
                std::dynamic_pointer_cast<fern::AttributeArgument>(
                    argument));
        BOOST_REQUIRE(attribute_argument);
        BOOST_REQUIRE_EQUAL(attribute_argument->data_type(),
            fern::DT_CONSTANT);
        BOOST_REQUIRE_EQUAL(attribute_argument->value_type(),
            fern::TypeTraits<T>::value_type);

        std::shared_ptr<fern::Attribute> const& attribute(
            attribute_argument->attribute());

        std::shared_ptr<fern::ConstantAttribute<T>>
            constant_attribute(std::dynamic_pointer_cast<
                fern::ConstantAttribute<T>>(attribute));
        BOOST_REQUIRE(constant_attribute);
        BOOST_CHECK_EQUAL(constant_attribute->values().value(), result);
    }
};


BOOST_FIXTURE_TEST_SUITE(interpreter, Support)

BOOST_AUTO_TEST_CASE(parse_string)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr vertex;

    // String with valid statements, should succeed.
    std::vector<fern::String> valid_statements = {
        "a = b + c",
        "a",
        "b + c",
        "",
        "# comment"
    };

    for(fern::String const& statement: valid_statements) {
        vertex = interpreter.parse_string(statement);
        BOOST_CHECK(vertex);
    }

    // String with invalid statement, syntax error with location information.
    try {
        interpreter.parse_string("a = b c");
        BOOST_CHECK(false);
    }
    catch(fern::ParseError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error parsing <string>:1:7:a = b c: invalid syntax");
    }
}


BOOST_AUTO_TEST_CASE(parse_file)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr vertex;

    // File with valid statement, should succeed.
    std::vector<fern::String> valid_files = {
        "valid-1.ran",
        "valid-2.ran"
    };

    for(fern::String const& filename: valid_files) {
        vertex = interpreter.parse_file(filename);
        BOOST_CHECK(vertex);
    }

    // File with invalid statement, syntax error with location information.
    try {
        interpreter.parse_file("invalid-1.ran");
        BOOST_CHECK(false);
    }
    catch(fern::ParseError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "Error parsing invalid-1.ran:1:7:a = b c: invalid syntax");
    }

    // Unreadable file, io error.
    try {
        interpreter.parse_file("valid-1_unreadable.ran");
        BOOST_CHECK(false);
    }
    catch(fern::IOError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error handling valid-1_unreadable.ran: Permission denied");
    }
}


BOOST_AUTO_TEST_CASE(validate)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr vertex;

    vertex = interpreter.parse_file("valid-1.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(fern::ValidateError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "valid-1.ran:3:4: Undefined identifier: b");
    }

    vertex = interpreter.parse_file("valid-2.ran");
    BOOST_CHECK(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(fern::ValidateError const& exception) {
        fern::String message = exception.message();
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
    catch(fern::ValidateError const& exception) {
        fern::String message = exception.message();
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
        // BOOST_CHECK_NO_THROW(interpreter.validate(vertex));
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
        catch(fern::ValidateError const& exception) {
            fern::String message = exception.message();
            // TODO Update message in test.
            // BOOST_CHECK_EQUAL(message,
            //     "<string>:1:4: Undefined operation: blah");
        }
    }

    // Operation with wrong number of arguments.
    vertex = interpreter.parse_string("abs(5, 6)");
    BOOST_REQUIRE(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(fern::ValidateError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "<string>:1:0: Wrong number of arguments for operation: abs: "
            "1 required, but 2 provided");
    }

    // Operation with wrong type of argument.
    vertex = interpreter.parse_string("abs(\"-5\")");
    BOOST_REQUIRE(vertex);

    try {
        interpreter.validate(vertex);
        BOOST_CHECK(false);
    }
    catch(fern::ValidateError const& exception) {
        fern::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "<string>:1:0: Wrong type of argument 1 provided for operation: "
            "abs: Constant|StaticField/Uint8|Int8|Uint16|Int16|Uint32|Int32|"
            "Uint64|Int64|Float32|Float64 required, but Constant/String "
            "provided");
    }
}


BOOST_AUTO_TEST_CASE(execute)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;

    // Calculate abs(-5) and leave the result on the stack for testing.
    {
        tree = interpreter.parse_string("abs(-5)");
        BOOST_REQUIRE(tree);
        interpreter.execute(tree);
        TestConstant<int64_t>()(interpreter, 5);
    }

    // TODO
    // // Calculate abs(-5) and leave the result on the stack for testing.
    // // Call a user defined function that does the calculation.
    // {
    //     tree = interpreter.parse_string(
    //         "def do_abs(value):\n"
    //         "    return abs(value)\n"
    //         "do_abs(-5)");
    //     BOOST_REQUIRE(tree);
    //     interpreter.execute(tree);
    //     TestConstant<int64_t>()(interpreter, 5);
    // }




    // fern::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // fern::Stack const& stack(execute_visitor.stack());

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

    // fern::ExecuteVisitor execute_visitor;
    // tree->Accept(execute_visitor);

    // fern::SymbolTable<boost::any> const& symbol_table(
    //     execute_visitor.symbol_table());
    // fern::Stack const& stack(execute_visitor.stack());

    // BOOST_CHECK_EQUAL(symbol_table.size(), 1u);
    // BOOST_CHECK(symbol_table.has_value("a"));
    // BOOST_CHECK(symbol_table.value("a").type() == typeid(int32_t));

    // BOOST_CHECK_EQUAL(stack.size(), 0u);
}


BOOST_AUTO_TEST_CASE(execute_read_with_raster_input)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;

    // Read feature.
    {
        interpreter.clear_stack();
        fern::String script = "read(\"raster-1.asc:raster-1\")";
        tree = interpreter.parse_string(script);
        interpreter.execute(tree);

        std::stack<std::shared_ptr<fern::Argument>> stack(
            interpreter.stack());
        BOOST_CHECK_EQUAL(stack.size(), 1u);

        std::shared_ptr<fern::Argument> const& argument(stack.top());
        BOOST_REQUIRE_EQUAL(argument->argument_type(),
            fern::ArgumentType::AT_FEATURE);

        std::shared_ptr<fern::FeatureArgument> const&
            feature_argument(
                std::dynamic_pointer_cast<fern::FeatureArgument>(
                    argument));
        BOOST_REQUIRE(feature_argument);

        std::shared_ptr<fern::Feature> const& feature(
            feature_argument->feature());

        BOOST_REQUIRE_EQUAL(feature->nr_attributes(), 1u);
        BOOST_REQUIRE(feature->contains_attribute("raster-1"));

        std::shared_ptr<fern::Attribute> const& attribute(
            feature->attribute("raster-1"));

        fern::FieldAttributePtr<int32_t> boxes_attribute(
            std::dynamic_pointer_cast<fern::FieldAttribute<int32_t>>(
                attribute));
        BOOST_REQUIRE(boxes_attribute);
        BOOST_REQUIRE_EQUAL(boxes_attribute->size(), 1u);

        fern::FieldValue<int32_t> const& value =
            *boxes_attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0],    -2);
        BOOST_CHECK_EQUAL(value[0][1],    -1);
        BOOST_CHECK_EQUAL(value[1][0],    -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0],     1);
        BOOST_CHECK_EQUAL(value[2][1],     2);
    }

    // Read attribute.
    {
        interpreter.clear_stack();
        fern::String script = "read(\"raster-1.asc:raster-1/raster-1\")";
        tree = interpreter.parse_string(script);
        interpreter.execute(tree);

        std::stack<std::shared_ptr<fern::Argument>> stack(
            interpreter.stack());
        BOOST_CHECK_EQUAL(stack.size(), 1u);

        std::shared_ptr<fern::Argument> const& argument(stack.top());
        BOOST_REQUIRE_EQUAL(argument->argument_type(),
            fern::ArgumentType::AT_ATTRIBUTE);

        std::shared_ptr<fern::AttributeArgument> const&
            attribute_argument(
                std::dynamic_pointer_cast<fern::AttributeArgument>(
                    argument));
        BOOST_REQUIRE(attribute_argument);

        std::shared_ptr<fern::Attribute> const& attribute(
            attribute_argument->attribute());

        fern::FieldAttributePtr<int32_t> boxes_attribute(
            std::dynamic_pointer_cast<fern::FieldAttribute<int32_t>>(
                attribute));
        BOOST_REQUIRE(boxes_attribute);
        BOOST_REQUIRE_EQUAL(boxes_attribute->size(), 1u);

        fern::FieldValue<int32_t> const& value =
            *boxes_attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0],    -2);
        BOOST_CHECK_EQUAL(value[0][1],    -1);
        BOOST_CHECK_EQUAL(value[1][0],    -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0],     1);
        BOOST_CHECK_EQUAL(value[2][1],     2);
    }

    /// Hier verder.
    /// {
    ///     fern::String script =
    ///         "feature = read(\"raster-1.asc\")\n"
    ///         "feature.raster-1";
    ///     tree = interpreter.parse_string(script);
    ///     interpreter.execute(tree);

    ///     std::stack<std::shared_ptr<fern::Argument>> stack(
    ///         interpreter.stack());
    ///     BOOST_CHECK_EQUAL(stack.size(), 1u);

    ///     std::shared_ptr<fern::Argument> const& argument(stack.top());
    ///     BOOST_REQUIRE_EQUAL(argument->argument_type(),
    ///         fern::ArgumentType::AT_FEATURE);

    ///     std::shared_ptr<fern::FeatureArgument> const&
    ///         feature_argument(
    ///             std::dynamic_pointer_cast<fern::FeatureArgument>(
    ///                 argument));
    ///     BOOST_REQUIRE(feature_argument);

    ///     std::shared_ptr<fern::Feature> const& feature(
    ///         feature_argument->feature());

    ///     BOOST_REQUIRE_EQUAL(feature->nr_attributes(), 1u);
    ///     BOOST_REQUIRE(feature->contains_attribute("raster-1"));

    ///     std::shared_ptr<fern::Attribute> const& attribute(
    ///         feature->attribute("raster-1"));

    ///     using Point = fern::Point<double, 2>;
    ///     using Box = fern::Box<Point>;
    ///     using BoxDomain = fern::SpatialDomain<Box>;
    ///     using Value = fern::ArrayValue<int32_t, 1>;
    ///     using ValuePtr = std::shared_ptr<Value>;
    ///     using BoxesAttribute = fern::SpatialAttribute<BoxDomain, ValuePtr>;

    ///     std::shared_ptr<BoxesAttribute> boxes_attribute(
    ///         std::dynamic_pointer_cast<BoxesAttribute>(attribute));
    ///     BOOST_REQUIRE(boxes_attribute);
    ///     BOOST_REQUIRE_EQUAL(boxes_attribute->size(), 1u);

    ///     ValuePtr value = boxes_attribute->values().cbegin()->second;
    ///     BOOST_REQUIRE_EQUAL(value->num_dimensions(), 1);
    ///     BOOST_REQUIRE_EQUAL(value->size(), 6);

    ///     BOOST_CHECK_EQUAL((*value)[0],    -2);
    ///     BOOST_CHECK_EQUAL((*value)[1],    -1);
    ///     BOOST_CHECK_EQUAL((*value)[2],    -0);
    ///     BOOST_CHECK_EQUAL((*value)[3], -9999);
    ///     BOOST_CHECK_EQUAL((*value)[4],     1);
    ///     BOOST_CHECK_EQUAL((*value)[5],     2);

    /// }
}


BOOST_AUTO_TEST_CASE(execute_abs_with_raster_input)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;

    {
        fern::String script =
            "abs(read(\"raster-1.asc:raster-1/raster-1\"))";
        tree = interpreter.parse_string(script);
        interpreter.execute(tree);

        std::stack<std::shared_ptr<fern::Argument>> stack(
            interpreter.stack());
        BOOST_CHECK_EQUAL(stack.size(), 1u);

        std::shared_ptr<fern::Argument> const& argument(stack.top());
        BOOST_REQUIRE_EQUAL(argument->argument_type(),
            fern::ArgumentType::AT_ATTRIBUTE);

        std::shared_ptr<fern::AttributeArgument> const&
            attribute_argument(
                std::dynamic_pointer_cast<fern::AttributeArgument>(
                    argument));
        BOOST_REQUIRE(attribute_argument);

        std::shared_ptr<fern::Attribute> const& attribute(
            attribute_argument->attribute());

        fern::FieldAttributePtr<int32_t> boxes_attribute(
            std::dynamic_pointer_cast<fern::FieldAttribute<int32_t>>(
                attribute));
        BOOST_REQUIRE(boxes_attribute);
        BOOST_REQUIRE_EQUAL(boxes_attribute->size(), 1u);

        fern::FieldValue<int32_t> const& value =
            *boxes_attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0], 2);
        BOOST_CHECK_EQUAL(value[0][1], 1);
        BOOST_CHECK_EQUAL(value[1][0], 0);
        // BOOST_CHECK_EQUAL(value[1][1], xxx);
        BOOST_CHECK_EQUAL(value[2][0],   1);
        BOOST_CHECK_EQUAL(value[2][1],   2);
    }
}


BOOST_AUTO_TEST_CASE(execute_write_with_raster_input)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;

    {
        // Read a raster, calculate the abs, write the result.
        // Read the new raster just written, leave it on the stack, and test
        // the values.
        fern::String script =
            "format = \"HFA\"\n"
            "attribute_name = \"output_dataset.map:output_dataset/output_dataset\"\n"
            "raster1 = read(\"raster-1.asc:raster-1/raster-1\")\n"
            "raster2 = abs(raster1)\n"
            "write(raster2, attribute_name, format)\n"
            "read(\"output_dataset.map:output_dataset/output_dataset\")\n"
            ;
        tree = interpreter.parse_string(script);
        interpreter.execute(tree);

        std::stack<std::shared_ptr<fern::Argument>> stack(
            interpreter.stack());
        BOOST_CHECK_EQUAL(stack.size(), 1u);

        std::shared_ptr<fern::Argument> const& argument(stack.top());
        BOOST_REQUIRE_EQUAL(argument->argument_type(),
            fern::ArgumentType::AT_ATTRIBUTE);

        std::shared_ptr<fern::AttributeArgument> const&
            attribute_argument(
                std::dynamic_pointer_cast<fern::AttributeArgument>(
                    argument));
        BOOST_REQUIRE(attribute_argument);

        std::shared_ptr<fern::Attribute> const& attribute(
            attribute_argument->attribute());

        fern::FieldAttributePtr<int32_t> boxes_attribute(
            std::dynamic_pointer_cast<fern::FieldAttribute<int32_t>>(
                attribute));
        BOOST_REQUIRE(boxes_attribute);
        BOOST_REQUIRE_EQUAL(boxes_attribute->size(), 1u);

        fern::FieldDomain const& domain(boxes_attribute->domain());
        BOOST_REQUIRE_EQUAL(domain.size(), 1u);
        fern::d2::Box const& box(domain.cbegin()->second);
        BOOST_CHECK_EQUAL(fern::get<0>(box.min_corner()), -1.0);
        BOOST_CHECK_EQUAL(fern::get<1>(box.min_corner()), -1.0);
        BOOST_CHECK_EQUAL(fern::get<0>(box.max_corner()), 1.0);
        BOOST_CHECK_EQUAL(fern::get<1>(box.max_corner()), 2.0);

        fern::FieldValue<int32_t> const& value(
            *boxes_attribute->values().cbegin()->second);
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0], 2);
        BOOST_CHECK_EQUAL(value[0][1], 1);
        BOOST_CHECK_EQUAL(value[1][0], 0);
        // BOOST_CHECK_EQUAL(value[1][1], xxx);
        BOOST_CHECK_EQUAL(value[2][0],   1);
        BOOST_CHECK_EQUAL(value[2][1],   2);
    }
}


BOOST_AUTO_TEST_CASE(execute_read_with_constant_input)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;

    {
        // Write a constant, read it again. Leave it on the stack, and test
        // the values.
        fern::String script = R"(
format = "Fern"
attribute_name = "earth.gnr:earth/gravity"
gravity = -9.8
write(gravity, attribute_name, format)
read(attribute_name))";
        tree = interpreter.parse_string(script);
        interpreter.execute(tree);
        TestConstant<double>()(interpreter, -9.8);
    }
}


BOOST_AUTO_TEST_CASE(execute_with_external_inputs)
{
    fern::Interpreter interpreter;
    fern::ModuleVertexPtr tree;
    fern::Interpreter::DataSyncSymbolTable data_sync_symbol_table;

    // If a script has undefined symbols, they must be provided from the
    // outside. After that, validation should result in fixed result expression
    // types and execution should succeed.

    // In this script, 'input' is undefined. It must be provided, otherwise
    // validation will throw an 'undefined identifier' exception.
    fern::String script = u8R"(
abs(input)
)";

    // Constant input.
    {
        std::shared_ptr<fern::DataSource> data_source(
            std::make_shared<fern::ConstantSource<int32_t>>(-9));
        fern::Interpreter::DataSourceSymbolTable data_source_symbol_table;
        data_source_symbol_table.push_scope();
        data_source_symbol_table.add_value("input", data_source);

        {
            tree = interpreter.parse_string(script);
            interpreter.clear_stack();
            BOOST_REQUIRE_NO_THROW(interpreter.validate(tree, data_source_symbol_table));
        }

        {
            tree = interpreter.parse_string(script);
            interpreter.clear_stack();
            BOOST_REQUIRE_NO_THROW(interpreter.execute(tree,
                data_source_symbol_table, data_sync_symbol_table));
            TestConstant<int32_t>()(interpreter, 9);
        }
    }

    // Constant input, from file.
    {
        std::shared_ptr<fern::DataSource> data_source(
            std::make_shared<fern::DatasetSource>(
                "constant-1.gnr:earth/gravity"));
        fern::Interpreter::DataSourceSymbolTable data_source_symbol_table;
        data_source_symbol_table.push_scope();
        data_source_symbol_table.add_value("input", data_source);

        {
            tree = interpreter.parse_string(script);
            interpreter.clear_stack();
            BOOST_REQUIRE_NO_THROW(interpreter.validate(tree, data_source_symbol_table));
        }

        {
            tree = interpreter.parse_string(script);
            interpreter.clear_stack();
            BOOST_REQUIRE_NO_THROW(interpreter.execute(tree,
                data_source_symbol_table, data_sync_symbol_table));
            TestConstant<double>()(interpreter, 9.8);
        }
    }

    // Make sure that data is read only once. An assertion will fail if the
    // same data is read more than once.
    {
        fern::String script = u8R"(
abs(input)
abs(input)
)";
        std::shared_ptr<fern::DataSource> data_source(
            std::make_shared<fern::ConstantSource<int32_t>>(-9));
        fern::Interpreter::DataSourceSymbolTable data_source_symbol_table;
        data_source_symbol_table.push_scope();
        data_source_symbol_table.add_value("input", data_source);

        tree = interpreter.parse_string(script);
        interpreter.clear_stack();
        BOOST_REQUIRE_NO_THROW(interpreter.execute(tree,
            data_source_symbol_table, data_sync_symbol_table));
    }
}


BOOST_AUTO_TEST_CASE(execute_with_external_outputs)
{
    using namespace fern;

    String script;
    Interpreter interpreter;

    // Outputs of a script can be coupled to a data sync. That way, data will
    // be saved.

    {
        Interpreter::DataSourceSymbolTable data_source_symbol_table;
        String dataset_pathname = "execute_with_external_outputs.gnr";
        std::shared_ptr<Dataset> dataset(open_dataset(dataset_pathname,
            OpenMode::OVERWRITE));
        std::shared_ptr<DataSync> data_sync_a(std::make_shared<DatasetSync>(
            dataset, "my_feature/a"));
        std::shared_ptr<DataSync> data_sync_b(std::make_shared<DatasetSync>(
            dataset, "my_feature/b"));
        Interpreter::DataSyncSymbolTable data_sync_symbol_table;
        data_sync_symbol_table.push_scope();
        data_sync_symbol_table.add_value("a", data_sync_a);
        data_sync_symbol_table.add_value("b", data_sync_b);

        {
            // Four potential outputs, one of which is redefined.
            interpreter.clear_stack();
            script = u8R"(
a = 5
b = 6.6
c = 7
a = 8
d = 9
)";
            BOOST_REQUIRE_NO_THROW(interpreter.execute(
                interpreter.parse_string(script),
                data_source_symbol_table, data_sync_symbol_table));

            // Read the output file and leave result on stack. Verify it is
            // indeed the value that should be written.
            interpreter.clear_stack();
            script = u8R"(
read("execute_with_external_outputs.gnr:my_feature/a")
)";
            BOOST_REQUIRE_NO_THROW(interpreter.execute(
                interpreter.parse_string(script)));
            TestConstant<int64_t>()(interpreter, 8);

            interpreter.clear_stack();
            script = u8R"(
read("execute_with_external_outputs.gnr:my_feature/b")
)";
            BOOST_REQUIRE_NO_THROW(interpreter.execute(
                interpreter.parse_string(script)));
            TestConstant<double>()(interpreter, 6.6);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
