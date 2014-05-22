#pragma once
#include "fern/algorithm/core/operation_categories.h"


namespace fern {

template<
    class Operation>
struct OperationTraits
{
    using category = typename Operation::category;
};

} // namespace fern
