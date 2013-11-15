#include "fern/command/interpreter.h"
#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <readline/readline.h>
#include <readline/history.h>
#include "fern/command/message.h"
#include "fern/core/exception.h"
#include "fern/core/type_traits.h"
#include "fern/core/value_type_traits.h"
#include "fern/feature/core/constant_attribute.h"
#include "fern/feature/core/feature.h"
#include "fern/feature/visitor/attribute_type_visitor.h"
#include "fern/interpreter/interpreter.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/operation/core/feature_argument.h"


namespace std {

template<>
struct default_delete<char>
{
    default_delete()=default;

    void operator()(char *buffer)
    {
       std::free(buffer);
    }
};

} // namespace std


namespace fern {

template<
    typename T>
inline void show_value(
    T const& value)
{
    static_assert(
        std::is_arithmetic<T>::value || std::is_same<T, String>::value,
        "T must be a number or a string");
    std::cout << value << "\n";
}


#define SHOW_VALUE_CASE(                                                       \
        value_type)                                                            \
    case value_type: {                                                         \
        typedef ValueTypeTraits<value_type>::type type;                        \
        ConstantAttribute<type> const& constant_attribute(                     \
            dynamic_cast<ConstantAttribute<type> const&>(*value));             \
        show_value<type>(constant_attribute.values().value());                 \
        break;                                                                 \
    }


void show_value(
    std::shared_ptr<Attribute> const& value)
{
    assert(value);

    fern::AttributeTypeVisitor visitor;
    value->Accept(visitor);

    switch(visitor.data_type()) {
        case DT_CONSTANT: {
            switch(visitor.value_type()) {
                SHOW_VALUE_CASE(VT_INT8)
                SHOW_VALUE_CASE(VT_INT16)
                SHOW_VALUE_CASE(VT_INT32)
                SHOW_VALUE_CASE(VT_INT64)
                SHOW_VALUE_CASE(VT_UINT8)
                SHOW_VALUE_CASE(VT_UINT16)
                SHOW_VALUE_CASE(VT_UINT32)
                SHOW_VALUE_CASE(VT_UINT64)
                SHOW_VALUE_CASE(VT_FLOAT32)
                SHOW_VALUE_CASE(VT_FLOAT64)
                SHOW_VALUE_CASE(VT_STRING)
            }

            break;
        }
        case DT_STATIC_FIELD: {
            std::cout << "static field" << std::endl;
            break;
        }
        // case DT_POLYGON: {
        //     std::cout << "polygon" << std::endl;
        //     break;
        // }
        default: {
            assert(false);
            break;
        }
    }
}

#undef SHOW_VALUE_CASE


void show_value(
    std::shared_ptr<AttributeArgument> const& value)
{
    assert(value);

    std::shared_ptr<Attribute> const& attribute(value->attribute());
    show_value(attribute);
}


void show_value(
    std::shared_ptr<FeatureArgument> const& value)
{
    assert(value);

    std::shared_ptr<Feature> const& feature(value->feature());
    std::cout << "feature containing " << feature->nr_attributes()
        << " attribute(s):" << "\n";

    for(auto attribute_name: feature->attribute_names()) {
        std::cout << attribute_name << "\n";
        show_value(feature->attribute(attribute_name));
    }
}


void show_value(
    std::shared_ptr<Argument> const& value)
{
    assert(value);

    switch(value->argument_type()) {
        case ArgumentType::AT_ATTRIBUTE: {
            show_value(std::dynamic_pointer_cast<AttributeArgument>(value));
            break;
        }
        case ArgumentType::AT_FEATURE: {
            show_value(std::dynamic_pointer_cast<FeatureArgument>(value));
            break;
        }
    }
}


void show_stack_values(
    std::stack<std::shared_ptr<Argument>> stack)
{
    while(!stack.empty()) {
        show_value(stack.top());
        stack.pop();
    }
}


void enter_interpreter()
{
    // 1. Create interpreter instance.
    // 2. Enter a loop:
    //    - Read a statement.
    //    - Execute the statement.
    // - Exit the loop if the user presses Ctrl-C, or exit(status).
    // - Implement exit(status) function which raises a SystemExit exception.
    // The interpreter represents an interactive script. Before statements
    // can be executed, a scope must be pushed.

    std::unique_ptr<char> line;
    fern::String statement;
    fern::Interpreter interpreter;
    std::shared_ptr<fern::ModuleVertex> script_vertex;

    fern::show_version();
    using_history();

    // Determine path name of history file. Reading the file will fail if it
    // doesn't exists, which is OK.
    std::string history_filename((
        fern::String::decode_from_default_encoding(std::getenv("HOME")) +
        "/.fern").encode_in_default_encoding());
    /* int result = */ read_history(history_filename.c_str());

    while(true) {
        // Obtain a statement from the user.
        std::unique_ptr<char> line(readline(">>> "));
        if(!line) {
            std::cout << std::endl;
            break;
        }

        add_history(line.get());
        statement = String::decode_from_default_encoding(line.get());

        try {
            // Parse and execute the statement.
            script_vertex = interpreter.parse_string(statement);
            interpreter.execute(script_vertex);

            // Print any values that are left on the stack and clear the stack.
            show_stack_values(interpreter.stack());
        }
        catch(fern::Exception const& exception) {
            fern::String message = exception.message();
            std::cerr << message << std::endl;
        }
        catch(std::exception const& exception) {
            std::cerr << "TODO: unhandled exception: "
                << fern::String(exception.what())
                << std::endl;
        }

        interpreter.clear_stack();
    }

    /* result = */ write_history(history_filename.c_str());
}

} // namespace fern
