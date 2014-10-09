#include "fern/back_end/core/back_end.h"


namespace fern {

BackEnd::BackEnd(
    OperationsPtr const& operations)

    : AstVisitor(),
      _operations(operations)

{
}


OperationsPtr const& BackEnd::operations() const
{
    return _operations;
}

} // namespace fern
