#include "ranally/language/execute_visitor.h"
#include "ranally/language/vertices.h"


namespace ranally {

void ExecuteVisitor::Visit(
    OperationVertex& vertex)
{
    // TODO Execute operation, given the arguments.
    // TODO First evaluate the arguments.
    // TODO Where to put the results?

    // TODO First print what needs to be done.
    // TODO Then think about how to do it.
    //      Maybe first execute, and afterwards think about where to store
    //      the values.
    //      Where to store the result values is related to where to store the
    //      argument values
    std::cout << "execute: " << vertex.name() << std::endl;
}

} // namespace ranally
