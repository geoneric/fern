#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/Vertices.h"


namespace ranally {

void ExecuteVisitor::Visit(
    OperationVertex& vertex)
{
    // TODO Execute operation, given the arguments.
    // 

    std::cout << "execute: " << vertex.name() << std::endl;
}

} // namespace ranally
