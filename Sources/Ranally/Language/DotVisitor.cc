#include "Ranally/Language/DotVisitor.h"
#include "Ranally/Language/Vertices.h"


namespace ranally {

DotVisitor::DotVisitor()

    : Visitor()

{
}


DotVisitor::~DotVisitor()
{
}


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
