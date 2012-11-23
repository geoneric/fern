#include "Ranally/Language/execute_visitor.h"
#include "Ranally/Language/vertices.h"


namespace ranally {

void ExecuteVisitor::Visit(
    OperationVertex& vertex)
{
    // TODO Execute operation, given the arguments.
    // 

    std::cout << "execute: " << vertex.name() << std::endl;
}

} // namespace ranally
