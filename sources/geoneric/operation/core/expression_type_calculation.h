#pragma once
#include <vector>
#include "geoneric/core/expression_type.h"


namespace geoneric {

class Operation;

ExpressionType standard_expression_type(Operation const& operation,
                                        size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types);

} // namespace geoneric
