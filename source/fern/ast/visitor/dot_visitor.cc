// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/ast/visitor/dot_visitor.h"
#include "fern/ast/core/vertices.h"


namespace fern {

String const& DotVisitor::script() const
{
    return _script;
}


void DotVisitor::set_script(
    String const& string)
{
    _script = string;
}


void DotVisitor::add_script(
    String const& string)
{
    _script += string;
}

} // namespace fern
