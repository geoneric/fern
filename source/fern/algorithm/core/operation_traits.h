#pragma once
#include "fern/algorithm/core/operation_categories.h"


namespace fern {
namespace algorithm {

template<
    typename Operation>
struct OperationTraits
{
    using category = typename Operation::category;
};

} // namespace algorithm
} // namespace fern
