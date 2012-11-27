#include "ranally/language/validate_visitor.h"
#include <stdexcept>
#include "ranally/core/string.h"
#include "ranally/operation/operation.h"
#include "ranally/language/vertices.h"


namespace ranally {

void ValidateVisitor::Visit(
    NameVertex& vertex)
{
    if(vertex.definitions().empty()) {
        // TODO Add to unit tests.
        throw std::runtime_error((boost::format(
            "%1%: undefined identifier")
                % vertex.name().encodeInUTF8()
            ).str().c_str());
    }
}


void ValidateVisitor::Visit(
    FunctionVertex& vertex)
{
    Visitor::Visit(vertex);

    // Find out if we know about an operation with the same name as this
    // function's name.
    if(!vertex.operation()) {
        throw std::runtime_error(("unknown function: " +
            vertex.name().encodeInUTF8()).c_str());
    }

    ranally::Operation const& operation(*vertex.operation());

    // Check if the number of arguments provided equals the required number of
    // arguments.
    if(vertex.expressions().size() < operation.parameters().size()) {
        // TODO Add to unit tests.
        throw std::runtime_error((boost::format(
            // <operation>: <description>: <details>
            "%1%: not enough arguments: %2% argument(s) required")
                % vertex.name().encodeInUTF8()
                % operation.parameters().size()
            ).str().c_str());
    }
    else if(vertex.expressions().size() > operation.parameters().size()) {
        // TODO Add to unit tests.
        throw std::runtime_error((boost::format(
            "%1%: too many arguments: %2% argument(s) required")
                % vertex.name().encodeInUTF8()
                % operation.parameters().size()
          ).str().c_str());
    }

    // TODO
    // // Check if the data type of each provided argument is accepted by the
    // // operation.
    // // ...

    // TODO
    // // Check if the value type of each provided argument is accepted by the
    // // operation.
    // // ...
}

} // namespace ranally
