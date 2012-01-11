#include "Ranally/Interpreter/Interpreter.h"
#include "Ranally/Operation/XmlParser.h"
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/ValidateVisitor.h"



namespace ranally {
namespace interpreter {

Interpreter::Interpreter()
{
}



Interpreter::~Interpreter()
{
}



void Interpreter::annotate(
  language::ScriptVertexPtr const& tree)
{
  // - Thread
  // - Identify
  // - Annotate

  language::ThreadVisitor threadVisitor;
  tree->Accept(threadVisitor);

  language::IdentifyVisitor identifyVisitor;
  tree->Accept(identifyVisitor);

  operation::OperationsPtr operations(operation::XmlParser().parse(
    operation::operationsXml));
  language::AnnotateVisitor annotateVisitor(operations);
  tree->Accept(annotateVisitor);
}



void Interpreter::validate(
  language::ScriptVertexPtr const& tree)
{
  // - Annotate
  // - Validate

  annotate(tree);
  language::ValidateVisitor validateVisitor;
  tree->Accept(validateVisitor);
}



void Interpreter::execute(
  language::ScriptVertexPtr const& tree)
{
  // validate
  // optimize
  // execute
  validate(tree);

  // TODO This can be done conditional.
  // language::OptimizeVisitor optimizeVisitor;
  // tree->Accept(optimizeVisitor);

  language::ExecuteVisitor executeVisitor;
  tree->Accept(executeVisitor);
}

} // namespace interpreter
} // namespace ranally

