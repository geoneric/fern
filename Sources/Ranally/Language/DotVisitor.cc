#include "Ranally/Language/DotVisitor.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include "Ranally/Language/Vertices.h"



namespace ranally {

DotVisitor::DotVisitor()

  // : /// _mode(Declaring) /// ,
    /// _definition(0)

{
}



DotVisitor::~DotVisitor()
{
}



UnicodeString const& DotVisitor::script() const
{
  return _script;
}



void DotVisitor::setScript(
  UnicodeString const& string)
{
  _script = string;
}



void DotVisitor::addScript(
  UnicodeString const& string)
{
  _script += string;
}



// void DotVisitor::setMode(
//   Mode mode)
// {
//   _mode = mode;
// }
// 
// 
// 
// DotVisitor::Mode DotVisitor::mode() const
// {
//   return _mode;
// }

} // namespace ranally

