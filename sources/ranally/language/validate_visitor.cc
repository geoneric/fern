#include "ranally/language/validate_visitor.h"
#include "ranally/core/exception.h"
#include "ranally/core/string.h"
#include "ranally/operation/operation.h"
#include "ranally/language/vertices.h"


namespace ranally {

void ValidateVisitor::Visit(
    NameVertex& vertex)
{
    if(vertex.definitions().empty()) {
        BOOST_THROW_EXCEPTION(detail::UndefinedIdentifier()
            << detail::ExceptionIdentifier(vertex.name())
            << detail::ExceptionLineNr(vertex.line())
            << detail::ExceptionColNr(vertex.col())
        );
    }
}


void ValidateVisitor::Visit(
    FunctionVertex& vertex)
{
    Visitor::Visit(vertex);

    // Find out if we know about an operation with the same name as this
    // function's name.
    if(!vertex.operation()) {
        BOOST_THROW_EXCEPTION(detail::UndefinedOperation()
            << detail::ExceptionFunction(vertex.name())
            << detail::ExceptionLineNr(vertex.line())
            << detail::ExceptionColNr(vertex.col())
        );
    }

    ranally::Operation const& operation(*vertex.operation());

    // Check if the number of arguments provided equals the required number of
    // arguments.
    if(vertex.expressions().size() != operation.parameters().size()) {
        // TODO Add to unit tests.
        BOOST_THROW_EXCEPTION(detail::ValidateError()
            << detail::ExceptionFunction(vertex.name())
            << detail::ExceptionRequiredNrArguments(
                operation.parameters().size())
            << detail::ExceptionProvidedNrArguments(
                vertex.expressions().size())
            << detail::ExceptionLineNr(vertex.line())
            << detail::ExceptionColNr(vertex.col())
        );
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
