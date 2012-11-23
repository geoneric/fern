#include "ranally/language/dot_visitor.h"
#include "ranally/language/vertices.h"


namespace ranally {

String const& DotVisitor::script() const
{
    return _script;
}


void DotVisitor::setScript(
    String const& string)
{
    _script = string;
}


void DotVisitor::addScript(
    String const& string)
{
    _script += string;
}

} // namespace ranally
