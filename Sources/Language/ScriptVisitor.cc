#include <iostream>

#include "ScriptVisitor.h"



namespace ranally {

ScriptVisitor::ScriptVisitor()
{
}



ScriptVisitor::~ScriptVisitor()
{
}



void ScriptVisitor::Visit(
  SyntaxVertex&)
{
  std::cout << "SyntaxVertex!" << std::endl;
}

} // namespace ranally

