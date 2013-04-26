#include "ranally/command/interpreter.h"
#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <readline/readline.h>
#include <readline/history.h>
#include "ranally/core/exception.h"
#include "ranally/command/message.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/interpreter/interpreter.h"
#include "ranally/operation/core/attribute_argument.h"
#include "ranally/operation/core/type_traits.h"
#include "ranally/operation/core/value_type_traits.h"


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


namespace ranally {

template<
    typename T>
inline void show_value(
    ScalarValue<T> const& value)
{
    std::cout << value() << "\n";
}


#define SHOW_VALUE_CASE(                                                       \
        value_type)                                                            \
    case value_type: {                                                         \
        typedef ValueTypeTraits<value_type>::type type;                        \
        std::shared_ptr<ScalarAttribute<type>> scalar_attribute(               \
            std::dynamic_pointer_cast<ScalarAttribute<type>>(attribute));      \
        show_value<type>(*scalar_attribute->value());                          \
        break;                                                                 \
    }

void show_value(
    std::shared_ptr<AttributeArgument> const& value)
{
    assert(value);

    std::shared_ptr<Attribute> const& attribute(value->attribute());

    switch(attribute->data_type()) {
        case DT_SCALAR: {
            switch(attribute->value_type()) {
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
        default: {
            assert(false);
            break;
        }
    }
}

#undef SHOW_VALUE_CASE


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
            assert(false);
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
    ranally::String statement;
    ranally::Interpreter interpreter;
    std::shared_ptr<ranally::ScriptVertex> script_vertex;

    ranally::show_version();
    using_history();

    // Determine path name of history file. Reading the file will fail if it
    // doesn't exists, which is OK.
    std::string history_filename((ranally::String::decode_from_default_encoding(
        std::getenv("HOME")) + "/.ranally").encode_in_default_encoding());
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
        catch(ranally::Exception const& exception) {
            ranally::String message = exception.message();
            std::cerr << message << std::endl;
        }
        catch(std::exception const& exception) {
            std::cerr << "TODO: unhandled exception: "
                << ranally::String(exception.what())
                << std::endl;
        }

        interpreter.clear_stack();
    }

    /* result = */ write_history(history_filename.c_str());
}

} // namespace ranally