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

    // Maybe create a hierarchy of execution blocks that share the same
    // base class, but are templated on the argument types, result types and
    // operation type. (It also includes control flow blocks.) Each execution
    // block is capable of returning its results in the correct types. Maybe
    // each execution block is a functor.
    // Arguments of operations then become execution blocks that can execute
    // themselves and return a result.
    // Can we use this idea for a compiler too?


    std::cout << "execute: " << vertex.name() << std::endl;
}

} // namespace ranally
