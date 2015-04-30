// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/back_end/core/back_end.h"


namespace fern {
namespace language {

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

} // namespace language
} // namespace fern
