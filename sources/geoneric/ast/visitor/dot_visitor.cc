#include "geoneric/ast/visitor/dot_visitor.h"
#include "geoneric/ast/core/vertices.h"


namespace geoneric {

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

} // namespace geoneric
