// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include "fern/core/expression_type.h"


namespace fern {
namespace language {

class Operation;

ExpressionType standard_expression_type(Operation const& operation,
                                        size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types);

} // namespace language
} // namespace fern
