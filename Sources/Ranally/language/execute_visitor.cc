#include "ranally/language/execute_visitor.h"
#include "ranally/language/vertices.h"


namespace ranally {

void ExecuteVisitor::Visit(
    OperationVertex& vertex)
{
    // TODO Execute operation, given the arguments.
    // 

    std::cout << "execute: " << vertex.name() << std::endl;
}

} // namespace ranally
