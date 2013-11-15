#pragma once
#include <vector>
#include "fern/core/expression_type.h"


namespace fern {

class Operation;

ExpressionType standard_expression_type(Operation const& operation,
                                        size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types);

} // namespace fern
