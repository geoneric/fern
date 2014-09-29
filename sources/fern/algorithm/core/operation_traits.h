#pragma once
#include "fern/algorithm/core/operation_categories.h"


namespace fern {
namespace algorithm {

template<
    class Operation>
struct OperationTraits
{
    using category = typename Operation::category;
};

} // namespace algorithm
} // namespace fern
