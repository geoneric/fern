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
