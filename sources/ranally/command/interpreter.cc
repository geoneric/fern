#include "ranally/command/interpreter.h"
#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <readline/readline.h>
#include <readline/history.h>
#include "ranally/core/exception.h"
#include "ranally/command/message.h"
#include "ranally/interpreter/interpreter.h"


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

void show_stack_values(
    std::stack<std::tuple<ResultType, boost::any>> stack)
{
    ResultType result_type;
    boost::any value;

    while(!stack.empty()) {
        std::tie(result_type, value) = stack.top();
        stack.pop();

        std::cout << result_type << std::endl;
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

    while(true) {
        // Obtain a statement from the user.
        std::unique_ptr<char> line(readline(">>> "));
        if(!line) {
            std::cout << std::endl;
            break;
        }

        add_history(line.get());
        statement = String(line.get());

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
            std::cerr << "TODO: unhandeled exception: " << exception.what()
                << std::endl;
        }

        interpreter.clear_stack();
    }
}

} // namespace ranally
