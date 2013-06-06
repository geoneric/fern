#include "ranally/back_end/core/back_end.h"


namespace ranally {

BackEnd::BackEnd(
    OperationsPtr const& operations)

    : _operations(operations)

{
}


OperationsPtr const& BackEnd::operations() const
{
    return _operations;
}

} // namespace ranally
