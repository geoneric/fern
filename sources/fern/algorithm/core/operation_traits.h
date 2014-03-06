#pragma once
#include "fern/algorithm/core/operation_categories.h"


namespace fern {

template<
    class Operation>
struct OperationTraits
{
    typedef typename Operation::category category;
};

} // namespace fern
